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
- Quick save with 's' or use 'w' to save via file dialog
- Open existing wells CSV with 'o' (try opening the example files, 6_well.csv and 8_well.csv)
- Undo changes with 'u'
- Output: `wells.csv` with columns: `well_name,center_x,center_y,radius,total_pixels`

### 2. Analyze Videos
```bash
python3 xen_loco_frame_stacker.py --wells-file wells.csv
```
- Processes all videos in `videos/` directory
- Uses well definitions from CSV created by well_editor_circular.py
- Generates coverage analysis for each well - tadpole trail images, percent total coverage and center coverage scores
- Output: `results/results.csv` containing percent total coverage and center coverage scores and cropped well images in grayscale or coloured by frame

### 3. Create Composite Visualization
```bash
python3 create_coverage_composite.py coverage
```
 - Creates two composite images of all wells sorted by both coverage metrics
 - Outputs (in `results/`):
	 - `percentage_total_coverage_composite.png` — wells ordered by total coverage percentage
	 - `center_coverage_score_composite.png` — wells ordered by center-weighted coverage score

## Well Editor Controls
- **Left-click**: select and drag well centers; drag well edge to resize ALL wells
- **'a'**: Add new well
- **'d'**: Delete selected well
- **'r'**: Auto-detect wells from current frame
- **'s'**: Quick save (calls the save flow)
- **'w'**: Save via file dialog (opens save dialog on most platforms)
- **'o'**: Open a wells CSV into the editor
- **'u'**: Undo last change
- **'h'**: Toggle help panel
- **Space**: Play/pause video
- **Left/Right arrows**: Navigate frames
- **ESC/Q**: Quit

For more information see WELL_EDITOR_README.md

