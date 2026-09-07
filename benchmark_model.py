#!/usr/bin/env python3
"""
Benchmark script for YOLOv8 pothole detection model
Measures inference speed on the selected local device.
"""

import time
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

def load_model(model_path):
    """Load the YOLOv8 model"""
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    return model


def resolve_device(requested):
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested

def create_test_image(width=640, height=640):
    """Create a test image for benchmarking"""
    # Create a test image with some patterns to simulate road/pothole scenarios
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    # Add some geometric shapes to make it more realistic
    cv2.rectangle(img, (50, 50), (200, 200), (100, 100, 100), -1)  # Gray rectangle
    cv2.circle(img, (400, 300), 50, (50, 50, 50), -1)  # Dark circle (simulating pothole)
    cv2.line(img, (0, 320), (640, 320), (100, 100, 100), 2)  # Horizontal line

    return img

def benchmark_model(model, image, device, num_warmup=10, num_iterations=100):
    """Benchmark model inference speed"""
    print(f"Warming up with {num_warmup} iterations...")

    # Warmup
    for _ in range(num_warmup):
        _ = model(image, device=device, verbose=False)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()

    print(f"Running benchmark with {num_iterations} iterations...")

    # Benchmark
    times = []
    for i in range(num_iterations):
        start_time = time.perf_counter()
        results = model(image, device=device, verbose=False)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        times.append(inference_time)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{num_iterations} iterations")

    # Calculate statistics
    times_array = np.array(times)
    avg_time = np.mean(times_array)
    std_time = np.std(times_array)
    min_time = np.min(times_array)
    max_time = np.max(times_array)
    fps = 1000.0 / avg_time

    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps,
        'times': times
    }

def get_model_info(model_path):
    """Get model file size and basic info"""
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return size_mb

def main():
    parser = argparse.ArgumentParser(description='Benchmark YOLOv8 pothole detection model')
    parser.add_argument('--model', type=str, default='inference/best.pt',
                       help='Path to the YOLOv8 model file')
    parser.add_argument('--image-size', type=int, default=640,
                       help='Input image size (square)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to run on (cpu, cuda, or auto)')

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 Pothole Detection Model Benchmark")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return 1

    # Display system info
    print(f"System Information:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"  Model Path: {args.model}")
    print(f"  Image Size: {args.image_size}x{args.image_size}")
    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1
    print(f"  Device: {device}")
    print()

    # Load model
    try:
        model = load_model(args.model)
        model_size = get_model_info(args.model)
        print(f"Model Size: {model_size:.2f} MB")
        print()
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Create test image
    print("Creating test image...")
    test_image = create_test_image(args.image_size, args.image_size)
    print("Test image created.")
    print()

    # Run benchmark
    try:
        results = benchmark_model(model, test_image, device, args.warmup, args.iterations)

        # Display results
        print("Benchmark Results:")
        print("-" * 30)
        print(f"Average Inference Time: {results['avg_time_ms']:.2f} ms")
        print(f"Standard Deviation:     {results['std_time_ms']:.2f} ms")
        print(f"Minimum Time:           {results['min_time_ms']:.2f} ms")
        print(f"Maximum Time:           {results['max_time_ms']:.2f} ms")
        print(f"Frames Per Second (FPS): {results['fps']:.2f}")
        print()

        # Save results to file
        results_file = "benchmark_results.txt"
        with open(results_file, 'w') as f:
            f.write("YOLOv8 Pothole Detection Model Benchmark Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Model Size: {model_size:.2f} MB\n")
            f.write(f"Image Size: {args.image_size}x{args.image_size}\n")
            f.write(f"Device: {device}\n")
            f.write(f"PyTorch Version: {torch.__version__}\n")
            f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
            f.write("\n")
            f.write("Benchmark Results:\n")
            f.write(f"  Average Inference Time: {results['avg_time_ms']:.2f} ms\n")
            f.write(f"  Standard Deviation:     {results['std_time_ms']:.2f} ms\n")
            f.write(f"  Minimum Time:           {results['min_time_ms']:.2f} ms\n")
            f.write(f"  Maximum Time:           {results['max_time_ms']:.2f} ms\n")
            f.write(f"  Frames Per Second (FPS): {results['fps']:.2f}\n")
            f.write("\n")

        print(f"Results saved to {results_file}")
        print()
        print("Benchmark completed successfully!")

        return 0

    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())