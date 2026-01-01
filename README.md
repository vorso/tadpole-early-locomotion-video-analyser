# tadpole-locomotion-video-analyser
Simple script for post-analysing Xenopus tadpole locomotion videos without XY data. Often early locomotion tracker software will render videos with crosshairs integrated into the video without producing a clean video to reanalyse. This python script thresholds and overlays each frame of the crosshair video to produce a trail using the position of the target in the video.

## Prerequisites

Requires Python3, OpenCV, and NumPy

## Quick Start Workflow

### 1. Define Well Positions (Interactive Editor)
```bash
python3 well_editor_circular.py path/to/video.avi
```
- Opens interactive GUI to define circular well positions
- Auto-detects wells or manually add/edit them
- Drag well centers to move, drag edges to resize all wells
- Press 's' to save configuration to CSV
- Output: `wells.csv` with columns: `well_name,center_x,center_y,radius,total_pixels`

### 2. Analyze Videos
```bash
python3 xen_loco_frame_stacker.py --wells-file wells.csv
```
- Processes all videos in `videos/` directory
- Uses well definitions from CSV
- Generates coverage analysis for each well
- Output: `results/results.csv` and cropped well images

### 3. Create Composite Visualization
```bash
python3 create_coverage_composite.py coverage
```
- Creates composite image of all wells sorted by coverage
- Optional: Use `center_coverage` instead of `coverage` for center-weighted sorting
- Output: `results/coverage_composite_coverage.png`

## Well Editor Controls
- **Mouse**: Click and drag well centers to move, drag edges to resize
- **'a'**: Add new well
- **'d'**: Delete selected well
- **'r'**: Auto-detect wells from current frame
- **'s'**: Save wells to CSV
- **'h'**: Toggle help panel
- **Space**: Play/pause video
- **Left/Right arrows**: Navigate frames
- **ESC/Q**: Quit
