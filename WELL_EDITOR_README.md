# Well Editor - Interactive Well Configuration Tool

The Well Editor is an interactive GUI tool for creating and editing well configurations for the tadpole locomotion video analysis pipeline.

## Features

- **Automatic Well Detection**: Detects circular wells from the first video frame
- **Visual Well Editing**: See wells overlaid on actual video frames
- **Video Playback**: Play through video with automatic looping
- **Live Frame Scrubbing**: Drag the frame slider to instantly see any frame
- **Separate Help Panel**: Toggleable side panel with controls (doesn't obscure video)
- **Drag & Drop**: Click and drag wells to reposition them
- **Resize Wells**: Drag corner handles to resize individual wells
- **Batch Operations**: Scale all wells simultaneously or snap them to the same size
- **Add/Remove Wells**: Dynamically add or remove wells as needed
- **CSV Import/Export**: Load existing configurations and save your changes

## Installation

The well editor requires the same dependencies as the main analysis script:
- Python 3.x
- OpenCV (cv2)
- NumPy

## Usage

### Basic Usage

```bash
# Auto-detect wells from video (recommended for new configurations)
python3 well_editor.py path/to/video.mp4

# Edit an existing well configuration
python3 well_editor.py path/to/video.mp4 --wells-csv 6_wells.csv
```

**Note**: When no CSV file is specified, the editor automatically detects circular wells from the first frame of the video. This works best for videos with light grey circles on a darker background.

### Examples

```bash
# Auto-detect wells from video (NEW!)
python3 well_editor.py videos/Xen_loco-20241114T114530-f30.avi

# Load and edit existing 6-well configuration
python3 well_editor.py videos/Xen_loco-20241114T114530-f30.avi --wells-csv 6_wells.csv

# Create a custom 4-well configuration
python3 well_editor.py videos/Xen_loco-20241114T114530-f30.avi --wells-csv 4_wells.csv
```

## Automatic Well Detection

When you open a video without specifying a CSV file, the editor automatically:

1. **Analyzes the first frame** to detect circular wells
2. **Identifies light grey circles** on the darker background
3. **Adds padding** around each detected circle (15 pixels) to ensure full coverage
4. **Prevents overlaps** by maintaining minimum distance between wells
5. **Sorts wells** logically (top to bottom, left to right)
6. **Names wells** sequentially (well1, well2, etc.)

The auto-detection uses OpenCV's HoughCircles algorithm optimized for:
- Circle radius: 50-150 pixels
- Minimum distance between circles: 100 pixels
- Light circles on dark backgrounds

If auto-detection fails or produces incorrect results, you can:
- Manually adjust the detected wells
- Add/remove wells as needed (A/D keys)
- Or load a pre-existing CSV configuration

## Controls

### Mouse Controls
- **Left Click on Well**: Select a well
- **Drag Well Center**: Move the well to a new position
- **Drag Well Corners**: Resize the well (blue circles at corners)

### Keyboard Controls
- **SPACE**: Play/Pause video playback (with automatic looping)
- **H**: Toggle help panel on/off
- **Left/Right Arrow**: Navigate to previous/next frame (pauses playback)
- **A**: Add a new well
- **D**: Delete the currently selected well
- **S**: Snap all wells to the same size as the selected well
- **+/=**: Scale all wells up by 10%
- **-/_**: Scale all wells down by 10%
- **W**: Save the current well configuration to CSV
- **Q or ESC**: Quit the editor

### Frame Navigation
- Use the **Frame trackbar** at the top of the window for live frame scrubbing
- Press **SPACE** to play through the video (loops automatically)
- Use **Left/Right arrow keys** for frame-by-frame navigation
- Trackbar updates in real-time as video plays

## Visual Indicators

- **Orange boxes**: Unselected wells
- **Green boxes**: Selected well
- **Blue circles**: Resize handles (appear on selected well)
- **Well labels**: Show well name and current size (e.g., "well1 (178x178)")
- **Status bar**: Shows current frame, total frames, well count, and play/pause status
- **Help panel**: Separate window with all controls (toggle with H key)

## Workflow

### For New Videos (with Auto-Detection):
1. **Load video** without CSV: `python3 well_editor.py video.avi`
2. **Review auto-detected wells** - they should cover all circular regions
3. **Adjust if needed** - drag to reposition, resize corners, or add/remove wells
4. **Verify** by playing through the video or scrubbing frames
5. **Save** your configuration with the W key

### For Existing Configurations:
1. **Load video with CSV**: `python3 well_editor.py video.avi --wells-csv wells.csv`
2. **Navigate** to a representative frame using the trackbar or play the video
3. **Adjust wells** by dragging them to the correct locations
4. **Resize wells** by dragging the corner handles
5. **Add/remove wells** as needed using A/D keys
6. **Verify** by playing through the video or scrubbing frames
7. **Save** your configuration with the W key

## Tips

- **Auto-detection works best** when wells are clearly visible circular regions with good contrast
- **Review detection results** - auto-detected wells may need minor adjustments
- Use video playback (SPACE) to see how wells align across all frames
- The frame slider provides instant feedback - drag it to quickly check different frames
- Use the snap feature (S key) to ensure all wells are the same size
- The +/- keys are useful for quickly adjusting all well sizes together
- Press H to hide the help panel for an unobstructed view
- Wells are saved with their exact pixel positions and sizes
- If auto-detection misses wells, use the A key to add them manually

## Output Format

The editor saves wells in CSV format compatible with `xen_loco_frame_stacker.py`:

```csv
well_name,center_x,center_y,size_x,size_y
well1,196,197,178,178
well2,373,197,178,178
...
```

## Integration with Analysis Pipeline

Once you've created a well configuration, use it with the main analysis script:

```bash
# Use your custom well configuration
python3 xen_loco_frame_stacker.py --wells-file my_custom_wells.csv
```

## Troubleshooting

**Video won't load**: Ensure the video file path is correct and the file format is supported by OpenCV (MP4, AVI, MOV, MKV)

**Can't see wells clearly**: Press H to hide the help panel for an unobstructed view

**Wells not saving**: Check that you have write permissions in the directory where you're trying to save the CSV file

**Playback too fast/slow**: The default is 30 FPS. Playback speed matches the video's natural frame rate

**Performance issues**: For very large videos, the editor may be slow. Consider using a shorter test video for well configuration.