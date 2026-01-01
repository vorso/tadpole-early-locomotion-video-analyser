from pathlib import Path
import cv2
import sys
import numpy as np
import csv
import re
import argparse
from typing import Set, Optional, List, Tuple

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}


# Calculate coverage weighted by distance from the center. Pixels closer
# to the center have higher weight. Returns a weighted coverage ratio.
def calculate_center_weighted_coverage(crop_bw):
    h, w = crop_bw.shape[:2]
    center_y, center_x = h / 2, w / 2

    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    max_distance = np.sqrt(center_x ** 2 + center_y ** 2)
    normalized_distances = distances / max_distance
    weights = 1.0 - normalized_distances

    # If input is single-channel grayscale, comparison works; if 3-channel BGR it still works per-channel
    black_mask = (crop_bw == 0).astype(float)
    # If crop_bw is 3-channel, reduce to a single-channel mask where all channels are black
    if black_mask.ndim == 3:
        black_mask = np.all(black_mask, axis=2).astype(float)

    weighted_coverage = np.sum(black_mask * weights)
    total_possible_weight = np.sum(weights)
    if total_possible_weight == 0:
        return 0.0
    return weighted_coverage / total_possible_weight


def load_wells_from_csv(csv_path: Path) -> Tuple[List[Tuple[str, int, int, int, int, int, int, int]], dict]:
    """
    Load well definitions from CSV file.
    Expected columns: well_name, center_x, center_y, radius, total_pixels
    Returns: (wells_list, total_pixels_dict)
    - wells_list: List of tuples (name, center_x, center_y, radius, x1, y1, x2, y2) - 8 elements
    - total_pixels_dict: Dict mapping well_name to total_pixels count
    """
    wells = []
    total_pixels_dict = {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['well_name']
            center_x = int(row['center_x'])
            center_y = int(row['center_y'])
            radius = int(row['radius'])

            # Get total_pixels (required column)
            total_pixels = int(row['total_pixels'])
            total_pixels_dict[name] = total_pixels

            # Calculate bounding box from circular well
            x1 = center_x - radius
            y1 = center_y - radius
            x2 = center_x + radius
            y2 = center_y + radius

            # Store: name, center_x, center_y, radius, x1, y1, x2, y2
            wells.append((name, center_x, center_y, radius, x1, y1, x2, y2))

    return wells, total_pixels_dict


def get_well_coverages_from_final_frame(frame, final_bw_frame, out_dir: Path, prefix: str,
                                        wells_data: List[Tuple[str, int, int, int, int, int, int, int]],
                                        total_pixels_dict: dict,
                                        ignored_wells: Optional[Set[int]] = None):
    """
    Returns two lists, one for coverage and one for center_coverage.
    Each list contains entries corresponding to each well.
    Ignored wells will have the string 'ignored_error' as their entry and
    will not have cropped images written out.

    wells_data: List of tuples (name, center_x, center_y, radius, x1, y1, x2, y2) defining well locations
    total_pixels_dict: Dict mapping well_name to total_pixels count
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]

    wells = wells_data

    # Output full frames
    out_file = out_dir / f"{prefix}-allwells-colour.png"
    cv2.imwrite(str(out_file), frame)
    print(f"Wrote {out_file}")

    out_file = out_dir / f"{prefix}-allwells-greyscale.png"
    cv2.imwrite(str(out_file), final_bw_frame)
    print(f"Wrote {out_file}")

    well_coverages = []
    well_center_coverages = []

    if ignored_wells is None:
        ignored_wells = set()

    for idx, (name, center_x, center_y, radius, x1, y1, x2, y2) in enumerate(wells, start=1):
        # If this well is marked ignored, append special value and skip writing crops
        if idx in ignored_wells:
            print(f"Skipping {name} for {prefix}: marked ignored")
            well_coverages.append("ignored_well")
            well_center_coverages.append("ignored_well")
            continue

        # clip to frame bounds
        x1c = max(0, min(w, x1))
        x2c = max(0, min(w, x2))
        y1c = max(0, min(h, y1))
        y2c = max(0, min(h, y2))

        if x2c <= x1c or y2c <= y1c:
            print(f"Skipping {name}: invalid crop after clipping ({x1c},{y1c})-({x2c},{y2c})", file=sys.stderr)
            well_coverages.append("ignored_well")
            well_center_coverages.append("ignored_well")
            continue

        crop = frame[y1c:y2c, x1c:x2c]
        crop_bw = final_bw_frame[y1c:y2c, x1c:x2c]

        # Create circular mask for the well
        crop_h, crop_w = crop_bw.shape[:2]
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # Calculate center position in cropped coordinates
        crop_center_x = center_x - x1c
        crop_center_y = center_y - y1c

        # Draw filled circle on mask
        cv2.circle(mask, (crop_center_x, crop_center_y), radius, 255, -1)

        # Apply mask to crops
        crop_masked = cv2.bitwise_and(crop, crop, mask=mask)
        crop_bw_masked = cv2.bitwise_and(crop_bw, crop_bw, mask=mask)

        # For greyscale output, set areas outside circle to white (255)
        crop_bw_output = crop_bw_masked.copy()
        crop_bw_output[mask == 0] = 255

        # Write out cropped images
        out_file = out_dir / f"{prefix}-{name}-colour.png"
        cv2.imwrite(str(out_file), crop_masked)

        out_file = out_dir / f"{prefix}-{name}-greyscale.png"
        cv2.imwrite(str(out_file), crop_bw_output)

        # Ensure single-channel for coverage calculations (use masked version, not output)
        if crop_bw_masked.ndim == 3:
            crop_bw_gray = cv2.cvtColor(crop_bw_masked, cv2.COLOR_BGR2GRAY)
        else:
            crop_bw_gray = crop_bw_masked

        # Calculate the total black pixels and coverage (only within circular mask)
        black_pixels = np.sum((crop_bw_gray == 0) & (mask == 255))

        # Get total possible pixels from CSV
        total_pixels = total_pixels_dict.get(name, mask.sum() // 255)

        # Calculate coverage percentage
        well_coverage = (black_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        # Calculate center-weighted coverage (only on masked region)
        # Apply mask to ensure we only calculate on circular region
        masked_crop = crop_bw_gray.copy()
        masked_crop[mask == 0] = 255  # Set non-circle pixels to white

        center_weighted_ratio = calculate_center_weighted_coverage(masked_crop)
        # Scale by total_pixels for percentage
        well_center_coverage = center_weighted_ratio * 100.0

        well_coverages.append(well_coverage)
        well_center_coverages.append(well_center_coverage)

        print(f"  {name} coverage: {well_coverage:.2f}%, center_coverage: {well_center_coverage:.2f}%")

    return well_coverages, well_center_coverages


def calculate_well_full_coverage_percentage(full_coverage_image_path: Path):
    img = cv2.imread(str(full_coverage_image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from {full_coverage_image_path}")
        return None, None

    total_pixels = img.shape[0] * img.shape[1]
    black_pixels = np.sum(img == 0)

    coverage_ratio = black_pixels / total_pixels if total_pixels > 0 else 0.0
    center_weighted_ratio = calculate_center_weighted_coverage(img)

    return coverage_ratio, center_weighted_ratio


def split_and_stack_frames(video_path: Path, out_dir: Path, wells_data: List[Tuple[str, int, int, int, int, int, int, int]],
                          total_pixels_dict: dict, ignored_wells: Optional[Set[int]] = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open {video_path}", file=sys.stderr)
        return None, None

    frame_count = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    SHOW = False
    if SHOW:
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Added", cv2.WINDOW_NORMAL)

    prev_frame = None
    prev_frame_bw = None

    while frame_count < length:
        ret, frame = cap.read()
        print(f"Processing frame number {frame_count} of {video_path.name}")

        if frame is None:
            break

        ret, thresholded_frame = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY)

        hue = int((frame_count / max(1, length - 1)) * 179)

        if thresholded_frame.ndim == 3:
            mask = cv2.cvtColor(thresholded_frame, cv2.COLOR_BGR2GRAY)
        else:
            mask = thresholded_frame

        hsv = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        color_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        color_frame = np.zeros_like(frame)
        white_mask = (mask == 255)
        color_frame[white_mask] = color_bgr[white_mask]

        if frame_count == 0:
            prev_frame = np.zeros_like(color_frame)
            prev_frame_bw = np.zeros_like(thresholded_frame)

        opacity = 0.0025
        added_frame = cv2.addWeighted(prev_frame, 1.0, color_frame, opacity, 0)
        added_frame_bw = cv2.addWeighted(prev_frame_bw, 1.0, thresholded_frame, opacity, 0)

        prev_frame = added_frame
        prev_frame_bw = added_frame_bw
        frame_count += 1

        if SHOW:
            cv2.imshow("Threshold", thresholded_frame)
            cv2.imshow("Added", added_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("User requested exit.", file=sys.stderr)
                break

    final_bw_frame = 255 - added_frame_bw

    if SHOW:
        cv2.imshow("Overlay", added_frame)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()

    return get_well_coverages_from_final_frame(added_frame, final_bw_frame, out_dir, video_path.stem, wells_data, total_pixels_dict, ignored_wells)


def load_ignored_wells(videos_dir: Path):
    """
    Load videos/ignore_these_wells.csv.
    Expected columns: video, ignored_wells
    ignored_wells can be like "1,2,3" or "1;2;3" etc.
    Returns dict mapping video_stem -> set(int wells)
    """
    ignored = {}
    csv_path = videos_dir / "ignore_these_wells.csv"
    if not csv_path.exists():
        return ignored

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row.get('video') or row.get('filename') or row.get('file')
            wells_field = (row.get('ignored_wells') or row.get('wells')
                           or row.get('well_numbers') or '')
            if not video_name:
                continue
            video_stem = Path(video_name).stem
            tokens = re.split(r'[\s,;|]+', wells_field.strip())
            nums = set()
            for t in tokens:
                if t.isdigit():
                    n = int(t)
                    if 1 <= n <= 6:
                        nums.add(n)
            if nums:
                ignored[video_stem] = nums
    return ignored


def _well_is_ignored(video_stem: str, well_num: int, csv_row: Optional[dict], ignored_map: dict):
    """
    Return True if the well should be ignored either via videos/ignore_these_wells.csv
    or because the corresponding coverage cell in csv_row == 'ignored_well' (case-insensitive).
    csv_row may be None if no results row is available at this point.
    """
    if ignored_map and video_stem in ignored_map and well_num in ignored_map[video_stem]:
        return True

    if csv_row:
        cov_key = f"well{well_num}_coverage"
        val = csv_row.get(cov_key, "")
        if isinstance(val, str) and val.strip().lower() == "ignored_well":
            return True

    return False


def main():
    parser = argparse.ArgumentParser(description='Process tadpole locomotion videos and analyze well coverage.')
    parser.add_argument('--wells-file', type=str, default='6_wells.csv',
                       help='Path to CSV file containing well definitions (default: 6_wells.csv)')
    args = parser.parse_args()

    base = Path(__file__).parent
    videos_dir = base / "videos"

    if not videos_dir.exists():
        print(f"No 'videos' folder found at {videos_dir}", file=sys.stderr)
        return

    # Load wells configuration
    wells_file = Path(args.wells_file)
    if not wells_file.is_absolute():
        wells_file = base / wells_file

    if not wells_file.exists():
        print(f"Wells file not found: {wells_file}", file=sys.stderr)
        return

    print(f"Loading well definitions from: {wells_file}")
    wells_data, total_pixels_dict = load_wells_from_csv(wells_file)
    print(f"Loaded {len(wells_data)} wells")
    if total_pixels_dict:
        print(f"Loaded total_pixels for {len(total_pixels_dict)} wells")

    results_root = base / "results"
    results_root.mkdir(exist_ok=True)

    found = False

    # No longer need global coverage ratios - using per-well total_pixels from CSV

    ignored_wells_map = load_ignored_wells(videos_dir)

    # Generate dynamic CSV header based on number of wells
    num_wells = len(wells_data)
    coverage_headers = [f"{wells_data[i][0]}_coverage" for i in range(num_wells)]
    center_coverage_headers = [f"{wells_data[i][0]}_center_coverage" for i in range(num_wells)]
    header = "video," + ",".join(coverage_headers) + "," + ",".join(center_coverage_headers) + "\n"

    with open(results_root / "results.csv", "w", buffering=1) as results_file:
        results_file.write(header)
        results_file.flush()

        for path in sorted(videos_dir.iterdir()):
            if not (path.is_file() and path.suffix in VIDEO_EXTS):
                continue
            found = True
            out_dir = results_root / path.stem
            print(f"Processing {path.name} -> {out_dir}")

            ignored_for_video = ignored_wells_map.get(path.stem, set())
            well_coverages, well_center_coverages = split_and_stack_frames(path, out_dir, wells_data, total_pixels_dict, ignored_for_video)

            # If function returned None, mark all wells as ignored
            if not well_coverages or not well_center_coverages:
                well_coverages = ["ignored_well"] * num_wells
                well_center_coverages = ["ignored_well"] * num_wells

            # Format entries: numeric -> formatted string, string -> pass through
            def fmt_list(lst):
                out = []
                for v in lst:
                    if isinstance(v, (float, int, np.floating, np.integer)):
                        out.append(f"{v:.2f}")
                    else:
                        out.append(str(v))
                return out

            coverage_list = fmt_list(well_coverages)
            center_list = fmt_list(well_center_coverages)

            coverage_str = ",".join(coverage_list)
            center_coverage_str = ",".join(center_list)

            results_file.write(f"{path.name},{coverage_str},{center_coverage_str}\n")
            results_file.flush()

        if not found:
            print("No video files found in 'videos' folder.")

if __name__ == "__main__":
    main()