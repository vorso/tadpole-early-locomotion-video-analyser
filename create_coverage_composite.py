#!/usr/bin/env python3
"""
Create a composite image of all well images sorted by coverage percentage.
Reads from frames/results.csv and combines individual well colour images.
"""

from pathlib import Path
import cv2
import numpy as np
import csv
import sys


def load_coverage_data(csv_path: Path, sort_by='coverage'):
    """
    Load coverage data from CSV and return sorted list of (video, well_num, coverage, center_coverage) tuples.

    Args:
        csv_path: Path to the CSV file
        sort_by: 'coverage' or 'center_coverage'
    """
    wells_data = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['video']
            # Remove extension from video name for matching with folder names
            video_stem = Path(video_name).stem

            # Extract coverage for each well
            for well_num in range(1, 7):
                coverage_key = f'well{well_num}_coverage'
                center_coverage_key = f'well{well_num}_center_coverage'

                if coverage_key in row:
                    coverage = float(row[coverage_key])
                    center_coverage = float(row.get(center_coverage_key, 0))
                    wells_data.append((video_stem, well_num, coverage, center_coverage))

    # Sort by specified metric (ascending)
    if sort_by == 'center_coverage':
        wells_data.sort(key=lambda x: x[3])
    else:
        wells_data.sort(key=lambda x: x[2])

    return wells_data


def create_composite_image(wells_data, frames_dir: Path, output_path: Path, sort_by='coverage', cols=10):
    """Create a composite image with all wells sorted by coverage."""

    # Load first image to get dimensions
    first_video, first_well, _, _ = wells_data[0]
    first_img_path = frames_dir / first_video / f"{first_video}-well{first_well}-colour.png"
    first_img = cv2.imread(str(first_img_path))

    if first_img is None:
        print(f"Error: Could not load first image from {first_img_path}")
        return

    well_height, well_width = first_img.shape[:2]

    # Calculate grid dimensions
    total_wells = len(wells_data)
    rows = (total_wells + cols - 1) // cols  # Ceiling division

    # Add space for text annotations
    text_height = 40
    cell_height = well_height + text_height
    cell_width = well_width

    # Create blank canvas
    canvas_height = rows * cell_height
    canvas_width = cols * cell_width
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # White background

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    text_color = (0, 0, 0)  # Black

    # Place each well image on the canvas
    for idx, (video_stem, well_num, coverage, center_coverage) in enumerate(wells_data):
        row = idx // cols
        col = idx % cols

        # Load well image
        img_path = frames_dir / video_stem / f"{video_stem}-well{well_num}-colour.png"
        well_img = cv2.imread(str(img_path))

        if well_img is None:
            print(f"Warning: Could not load {img_path}")
            continue

        # Calculate position
        y_start = row * cell_height
        x_start = col * cell_width

        # Place image
        canvas[y_start:y_start + well_height, x_start:x_start + well_width] = well_img

        # Add text annotation
        text_y = y_start + well_height + 15
        text_x = x_start + 5

        # Line 1: Video name (truncated if needed)
        video_text = video_stem[:25] if len(video_stem) > 25 else video_stem
        cv2.putText(canvas, video_text, (text_x, text_y),
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Line 2: Well number and coverage
        if sort_by == 'center_coverage':
            info_text = f"Well {well_num}: C{center_coverage:.1f}%"
        else:
            info_text = f"Well {well_num}: {coverage:.1f}%"
        cv2.putText(canvas, info_text, (text_x, text_y + 15),
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Save composite image
    cv2.imwrite(str(output_path), canvas)
    print(f"Created composite image: {output_path}")
    print(f"Total wells: {total_wells}")
    print(f"Grid size: {rows} rows x {cols} columns")
    print(f"Image size: {canvas_width}x{canvas_height} pixels")


def main():
    base = Path(__file__).parent
    csv_path = base / "results" / "results.csv"
    frames_dir = base / "results"

    # Check for command line argument
    sort_by = 'coverage'
    if len(sys.argv) > 1:
        if sys.argv[1] in ['coverage', 'center_coverage']:
            sort_by = sys.argv[1]
        else:
            print(f"Invalid sort option: {sys.argv[1]}")
            print("Usage: python create_coverage_composite.py [coverage|center_coverage]")
            return

    output_filename = f"coverage_composite_{sort_by}.png"
    output_path = base / "results" / output_filename

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading coverage data from CSV (sorting by {sort_by})...")
    wells_data = load_coverage_data(csv_path, sort_by=sort_by)

    if not wells_data:
        print("No well data found in CSV")
        return

    print(f"Found {len(wells_data)} wells")
    if sort_by == 'center_coverage':
        print(f"Center coverage range: {wells_data[0][3]:.1f}% to {wells_data[-1][3]:.1f}%")
    else:
        print(f"Coverage range: {wells_data[0][2]:.1f}% to {wells_data[-1][2]:.1f}%")

    print(f"\nCreating composite image sorted by {sort_by}...")
    create_composite_image(wells_data, frames_dir, output_path, sort_by=sort_by, cols=10)

    print("\nDone!")


if __name__ == "__main__":
    main()

