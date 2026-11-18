#!/usr/bin/env python3
"""
Create Figure 1: Pipeline Architecture Diagram
Four-thread pipeline: Producer → Inference → Encoder → DB → FastAPI Dashboard
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
colors = {
    'producer': '#4CAF50',      # Green
    'inference': '#2196F3',     # Blue
    'encoder': '#FF9800',       # Orange
    'db': '#9C27B0',            # Purple
    'api': '#F44336',           # Red
    'browser': '#607D8B',       # Blue Grey
    'arrow': '#333333',
    'text': '#212121',
    'bg': '#FAFAFA'
}

# Box style
box_style = dict(boxstyle="round,pad=0.5", facecolor=colors['bg'], edgecolor=colors['arrow'], linewidth=2)

# Thread boxes (vertical strips)
thread_boxes = [
    (0.5, 0.5, 2.5, 5, colors['producer'], 'Producer Thread\n(Decode + Preprocess)\n\n• cv2.VideoCapture\n• Letterbox resize\n• Normalize [0,1]\n• HWC→CHW\n• Pin host memory'),
    (3.2, 0.5, 2.8, 5, colors['inference'], 'Inference Thread\n(TensorRT INT8)\n\n• CUDA context push/pop\n• Async H2D/D2H copies\n• TRT execute_v2()\n• Class-aware NMS\n• BoxTracker (IoU, TTL=30)\n• Draw overlays'),
    (6.3, 0.5, 2.5, 5, colors['encoder'], 'Encoder Thread\n(JPEG for MJPEG)\n\n• cv2.imencode()\n• ~8.9 ms/frame\n• 2-slot drop-oldest queue\n• OpenCV releases GIL\n• Offloaded from inference'),
    (9.1, 0.5, 2.5, 5, colors['db'], 'DB Worker Thread\n(SQLite Events)\n\n• INSERT events\n• Batch commits\n• Snapshot management\n• /uploads/ frame storage\n• ISO8601 timestamps'),
]

for x, y, w, h, color, label in thread_boxes:
    # Colored header bar
    header = FancyBboxPatch((x, y+h-0.8), w, 0.8, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor=color, linewidth=0)
    ax.add_patch(header)
    # Main box
    box = FancyBboxPatch((x, y), w, h-0.8, **box_style)
    ax.add_patch(box)
    # Label
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=10, color=colors['text'], fontfamily='monospace')

# FastAPI Dashboard box
dash_x, dash_y, dash_w, dash_h = 0.5, 5.2, 11.1, 1.5
dash_box = FancyBboxPatch((dash_x, dash_y), dash_w, dash_h, boxstyle="round,pad=0.3",
                          facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
ax.add_patch(dash_box)
ax.text(dash_x + dash_w/2, dash_y + dash_h - 0.3, 'FastAPI Dashboard (Port 8000)',
        ha='center', va='top', fontsize=12, fontweight='bold', color='#1565C0')
dash_features = 'GET /health, /local-status, /video-feed (MJPEG), /video-feed-stats, /potholes, /api/gnss, /api/frames, /api/export'
ax.text(dash_x + dash_w/2, dash_y + 0.5, dash_features, ha='center', va='center',
        fontsize=9, color=colors['text'], fontfamily='monospace')

# Browser box
browser_x, browser_y, browser_w, browser_h = 11.9, 5.2, 1.6, 1.5
browser_box = FancyBboxPatch((browser_x, browser_y), browser_w, browser_h, boxstyle="round,pad=0.3",
                             facecolor='#ECEFF1', edgecolor='#546E7A', linewidth=2)
ax.add_patch(browser_box)
ax.text(browser_x + browser_w/2, browser_y + browser_h/2, 'Browser\n\n• Live Feed\n• GIS Map\n• Events Table\n• Annotate\n• Train\n• Timeline',
        ha='center', va='center', fontsize=9, color=colors['text'], fontfamily='monospace')

# Arrows between threads
arrow_props = dict(arrowstyle='->', color=colors['arrow'], lw=2, shrinkA=5, shrinkB=5)
thread_centers_y = 3.0
centers = [(3.0, thread_centers_y), (6.0, thread_centers_y), (8.8, thread_centers_y), (10.35, thread_centers_y)]
for i in range(len(centers)-1):
    ax.annotate('', xy=centers[i+1], xytext=centers[i], arrowprops=arrow_props)

# Arrow from DB to Dashboard
ax.annotate('', xy=(6.05, 5.2), xytext=(10.35, 5.5), arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2,
                                                                      connectionstyle="arc3,rad=-0.3"))

# Arrow from Dashboard to Browser
ax.annotate('', xy=(11.9, 5.95), xytext=(11.6, 5.95), arrowprops=arrow_props)

# Input arrow (Camera/Video)
ax.annotate('', xy=(1.75, 5.5), xytext=(0.5, 5.5), arrowprops=dict(arrowstyle='<-', color=colors['arrow'], lw=2))
ax.text(1.75, 5.8, 'Camera / Video', ha='center', va='bottom', fontsize=10, fontweight='bold', color=colors['text'])

# Queue labels between threads
queue_labels = [
    (3.0, 3.3, 'Queue\n(maxsize=2)'),
    (6.0, 3.3, 'Queue\n(maxsize=2)'),
    (8.8, 3.3, 'Queue\n(maxsize=2)'),
]
for x, y, label in queue_labels:
    ax.text(x, y, label, ha='center', va='center', fontsize=8, color='#757575',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#FFF9C4', edgecolor='#FBC02D', linewidth=1))

# Title
ax.text(7, 6.5, 'Figure 1: Four-Thread Pipeline Architecture', ha='center', va='center',
        fontsize=14, fontweight='bold', color=colors['text'])

# Legend
legend_elements = [
    mpatches.Patch(color=colors['producer'], label='Producer Thread'),
    mpatches.Patch(color=colors['inference'], label='Inference Thread (TensorRT INT8)'),
    mpatches.Patch(color=colors['encoder'], label='Encoder Thread (JPEG)'),
    mpatches.Patch(color=colors['db'], label='DB Worker Thread (SQLite)'),
    mpatches.Patch(color='#1976D2', label='FastAPI Dashboard'),
    mpatches.Patch(color='#546E7A', label='Browser Client'),
]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.08),
          ncol=3, fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('/home/edgeai/Documents/Srikaanth CDAC/CDAC-pothole/charts/00_pipeline_architecture.png', dpi=300, bbox_inches='tight')
print("Figure 1 saved to charts/00_pipeline_architecture.png")