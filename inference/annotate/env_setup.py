"""Runtime environment fixups for pip-installed CUDA libraries.

torch 2.11 links against ``libcudss.so.0``, which pip installs under
``site-packages/nvidia/cu12/lib``. That directory is not on the loader
search path on JetPack, so ``import torch`` raises::

    ImportError: libcudss.so.0: cannot open shared object file

The glibc dynamic loader reads ``LD_LIBRARY_PATH`` only once, at process
start, so mutating ``os.environ`` at runtime has no effect. The only
reliable fix is to re-exec the process with the corrected environment.

A second Jetson-specific issue is handled here: torch 2.11 uses NVML for
CUDA error checks by default. NVML is only partially supported on the
Tegra iGPU (``nvidia-smi`` reports "Not Supported" for several fields),
so the CUDA caching allocator trips::

    RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
    "/opt/pytorch/c10/cuda/CUDACachingAllocator.cpp":1165

Setting ``PYTORCH_NVML_BASED_CUDA_CHECK=0`` makes torch fall back to the
CUDA runtime for those checks, which works correctly on Tegra.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_SENTINEL = "CDAC_CUDA_LIBS_FIXED"

#: Torch must not use NVML for CUDA checks on the Tegra iGPU.
TORCH_ENV = {"PYTORCH_NVML_BASED_CUDA_CHECK": "0"}


def cuda_lib_dirs() -> list[Path]:
    """Locate pip-installed NVIDIA CUDA library directories."""
    candidates: list[Path] = []
    for base in list(site.getsitepackages() or []) + [site.getusersitepackages()]:
        if base:
            candidates.append(Path(base) / "nvidia" / "cu12" / "lib")
            candidates.append(Path(base) / "nvidia" / "lib")
    seen: set[str] = set()
    found: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            found.append(path)
    return found


def ensure_cuda_libs() -> None:
    """Re-exec the current process with LD_LIBRARY_PATH set for CUDA libs.

    Safe to call at the start of ``main()``. Does nothing when the
    libraries are already reachable or when the process has already been
    re-executed.
    """
    if os.environ.get(_SENTINEL) == "1":
        return
    needed = [str(d) for d in cuda_lib_dirs()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = current.split(":") if current else []
    missing = [item for item in needed if item not in parts]
    if not missing and os.environ.get("PYTORCH_NVML_BASED_CUDA_CHECK") == TORCH_ENV["PYTORCH_NVML_BASED_CUDA_CHECK"]:
        return
    env = dict(os.environ)
    if missing:
        env["LD_LIBRARY_PATH"] = ":".join(missing + parts)
    env.update(TORCH_ENV)
    env[_SENTINEL] = "1"
    # orig_argv keeps the original invocation form (script, -m module, -c code)
    argv = list(getattr(sys, "orig_argv", None) or sys.argv)
    try:
        os.execve(sys.executable, argv, env)
    except OSError:  # pragma: no cover - fall back to running without the fix
        pass


def cuda_lib_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment dict with the CUDA library paths applied.

    Used when spawning subprocesses (for example the training pipeline)
    so the child process resolves ``libcudss.so.0`` on startup and uses
    the CUDA runtime instead of NVML for device checks.
    """
    env = dict(base_env if base_env is not None else os.environ)
    needed = [str(d) for d in cuda_lib_dirs()]
    if needed:
        current = env.get("LD_LIBRARY_PATH", "")
        parts = current.split(":") if current else []
        missing = [item for item in needed if item not in parts]
        if missing:
            env["LD_LIBRARY_PATH"] = ":".join(missing + parts)
    env.update(TORCH_ENV)
    return env
