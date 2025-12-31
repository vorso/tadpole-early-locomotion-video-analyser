from pathlib import Path
import cv2
import sys
import numpy as np
import csv
import re
from typing import Set, Optional

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


def get_well_coverages_from_final_frame(frame, final_bw_frame, out_dir: Path, prefix: str,
                                        ignored_wells: Optional[Set[int]] = None):
    """
    Returns two lists, one for coverage and one for center_coverage.
    Each list contains exactly 6 entries corresponding to wells 1..6.
    Ignored wells will have the string 'ignored_error' as their entry and
    will not have cropped images written out.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]

    # Hardcoded well locations by centre
    w1 = (196, 197)
    w2 = (373, 197)
    w3 = (550, 197)
    w4 = (196, 372)
    w5 = (373, 372)
    w6 = (550, 372)

    # Hardcoded well sizes
    sx = int(178 / 2)  # well width half
    sy = int(178 / 2)  # well height half

    # Build well list as (name, x1, y1, x2, y2)
    wells = [
        ("well1", w1[0] - sx, w1[1] - sy, w1[0] + sx, w1[1] + sy),
        ("well2", w2[0] - sx, w2[1] - sy, w2[0] + sx, w2[1] + sy),
        ("well3", w3[0] - sx, w3[1] - sy, w3[0] + sx, w3[1] + sy),
        ("well4", w4[0] - sx, w4[1] - sy, w4[0] + sx, w4[1] + sy),
        ("well5", w5[0] - sx, w5[1] - sy, w5[0] + sx, w5[1] + sy),
        ("well6", w6[0] - sx, w6[1] - sy, w6[0] + sx, w6[1] + sy)
    ]

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

    for idx, (name, x1, y1, x2, y2) in enumerate(wells, start=1):
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

        # Write out cropped images
        out_file = out_dir / f"{prefix}-{name}-colour.png"
        cv2.imwrite(str(out_file), crop)

        out_file = out_dir / f"{prefix}-{name}-greyscale.png"
        cv2.imwrite(str(out_file), crop_bw)

        # Ensure single-channel for coverage calculations
        if crop_bw.ndim == 3:
            crop_bw_gray = cv2.cvtColor(crop_bw, cv2.COLOR_BGR2GRAY)
        else:
            crop_bw_gray = crop_bw

        # Calculate the total black pixels and coverage
        total_pixels = crop_bw_gray.shape[0] * crop_bw_gray.shape[1]
        black_pixels = np.sum(crop_bw_gray == 0)
        frame_coverage_ratio = black_pixels / total_pixels if total_pixels > 0 else 0.0

        # highest_coverage_ratio expected to be set globally
        try:
            well_coverage = (frame_coverage_ratio / highest_coverage_ratio) * 100.0
        except Exception:
            well_coverage = 0.0

        center_weighted_ratio = calculate_center_weighted_coverage(crop_bw_gray)
        try:
            well_center_coverage = (center_weighted_ratio / highest_center_coverage_ratio) * 100.0
        except Exception:
            well_center_coverage = 0.0

        well_coverages.append(well_coverage)
        well_center_coverages.append(well_center_coverage)

        print(f"  {name} coverage: {well_coverage:.2f}%, center_coverage: {well_center_coverage:.2f}%")

    # Ensure lists are length 6
    while len(well_coverages) < 6:
        well_coverages.append("ignored_well")
    while len(well_center_coverages) < 6:
        well_center_coverages.append("ignored_well")

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


def split_and_stack_frames(video_path: Path, out_dir: Path, ignored_wells: Optional[Set[int]] = None):
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

    return get_well_coverages_from_final_frame(added_frame, final_bw_frame, out_dir, video_path.stem, ignored_wells)


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
    base = Path(__file__).parent
    videos_dir = base / "videos"

    if not videos_dir.exists():
        print(f"No 'videos' folder found at {videos_dir}", file=sys.stderr)
        return

    results_root = base / "results"
    results_root.mkdir(exist_ok=True)

    found = False

    global highest_coverage_ratio, highest_center_coverage_ratio
    highest_coverage_ratio, highest_center_coverage_ratio = calculate_well_full_coverage_percentage(Path("well_full_coverage.png"))
    # fallback to 1.0 to avoid division by zero if load failed
    if not highest_coverage_ratio or highest_coverage_ratio == 0:
        highest_coverage_ratio = 1.0
    if not highest_center_coverage_ratio or highest_center_coverage_ratio == 0:
        highest_center_coverage_ratio = 1.0

    ignored_wells_map = load_ignored_wells(videos_dir)

    header = ("video,well1_coverage,well2_coverage,well3_coverage,well4_coverage,well5_coverage,well6_coverage,"
              "well1_center_coverage,well2_center_coverage,well3_center_coverage,well4_center_coverage,well5_center_coverage,well6_center_coverage\n")

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
            well_coverages, well_center_coverages = split_and_stack_frames(path, out_dir, ignored_for_video)

            # Ensure we always have 6 entries; if function returned None, mark all ignored_well
            if not well_coverages or not well_center_coverages:
                well_coverages = ["ignored_well"] * 6
                well_center_coverages = ["ignored_well"] * 6

            # Format entries: numeric -> formatted string, string -> pass through
            def fmt_list(lst):
                out = []
                for v in lst:
                    if isinstance(v, (float, int, np.floating, np.integer)):
                        out.append(f"{v:.2f}")
                    else:
                        out.append(str(v))
                # pad/truncate to 6
                out = out[:6] + ["ignored_well"] * max(0, 6 - len(out))
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