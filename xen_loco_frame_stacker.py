from pathlib import Path
import cv2
import sys
import numpy as np

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}

"""
Calculate coverage weighted by distance from the center. Pixels closer
to the center have higher weight. Returns a weighted coverage ratio.
"""
def calculate_center_weighted_coverage(crop_bw):
    h, w = crop_bw.shape[:2]
    center_y, center_x = h / 2, w / 2

    # Create distance map from center
    y_coords, x_coords = np.ogrid[:h, :w]
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)

    # Normalize distances to 0-1 range (0 at center, 1 at furthest corner)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    normalized_distances = distances / max_distance

    # Create weight map: higher weight at center (invert normalized distances)
    # Weight ranges from 1.0 at center to 0.0 at edges
    weights = 1.0 - normalized_distances

    # Apply weights only to black pixels (coverage)
    black_mask = (crop_bw == 0).astype(float)
    weighted_coverage = np.sum(black_mask * weights)

    # Normalize by total possible weighted coverage (if all pixels were black)
    total_possible_weight = np.sum(weights)

    return weighted_coverage / total_possible_weight

"""
Once the blended frame is complete, we can extract the coverage from
the black/white final frame for each well region. We can also write
out cropped images for each well and final colour and greyscale frames.
"""
def get_well_coverages_from_final_frame(frame, final_bw_frame, out_dir: Path, prefix: str):

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
    sx = int(178 / 2) # well width
    sy = int(178 / 2)# well height

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

    for name, x1, y1, x2, y2 in wells:
        # clip to frame bounds
        x1c = max(0, min(w, x1))
        x2c = max(0, min(w, x2))
        y1c = max(0, min(h, y1))
        y2c = max(0, min(h, y2))

        if x2c <= x1c or y2c <= y1c:
            # skip invalid / out-of-bounds crop
            print(f"Skipping {name}: invalid crop after clipping ({x1c},{y1c})-({x2c},{y2c})", file=sys.stderr)
            continue

        crop = frame[y1c:y2c, x1c:x2c]
        crop_bw = final_bw_frame[y1c:y2c, x1c:x2c]

        # Write out cropped images
        out_file = out_dir / f"{prefix}-{name}-colour.png"
        cv2.imwrite(str(out_file), crop)

        out_file = out_dir / f"{prefix}-{name}-greyscale.png"
        cv2.imwrite(str(out_file), crop_bw)

        # Ensure crop_bw is single channel
        crop_bw = cv2.cvtColor(crop_bw, cv2.COLOR_BGR2GRAY)

        # Calculate the total black pixels and coverage
        total_pixels = crop_bw.shape[0] * crop_bw.shape[1]
        black_pixels = np.sum(crop_bw == 0)

        frame_coverage_ratio = black_pixels / total_pixels
        well_coverage = (frame_coverage_ratio / highest_coverage_ratio) * 100.0

        # Calculate center-weighted coverage
        center_weighted_ratio = calculate_center_weighted_coverage(crop_bw)
        well_center_coverage = (center_weighted_ratio / highest_center_coverage_ratio) * 100.0

        well_coverages.append(well_coverage)
        well_center_coverages.append(well_center_coverage)

        print(f"  {name} coverage: {well_coverage:.2f}%, center_coverage: {well_center_coverage:.2f}%")

    return well_coverages, well_center_coverages

"""
Calculate the maximum possible coverage from the well_full_coverage.png
image.
"""
def calculate_well_full_coverage_percentage(full_coverage_image_path: Path):

    img = cv2.imread(full_coverage_image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from {full_coverage_image_path}")
        return None, None

    total_pixels = img.shape[0] * img.shape[1]
    black_pixels = np.sum(img == 0)

    # Calculate standard coverage ratio
    coverage_ratio = black_pixels / total_pixels

    # Calculate center-weighted coverage ratio
    center_weighted_ratio = calculate_center_weighted_coverage(img)

    return coverage_ratio, center_weighted_ratio

"""
Split video into frames and stack them, then write out the resulting
well coverage data
"""
def split_and_stack_frames(video_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Failed to open {video_path}", file=sys.stderr)
        return

    frame_count = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Can be helpful for debugging to show the frames as they are
    # being processed. Set to True to enable.
    SHOW = False

    if SHOW:
        cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Added", cv2.WINDOW_NORMAL)

    while frame_count < length:
        ret, frame = cap.read()

        print(f"Processing frame number {frame_count} of {video_path.name}")

        if frame is None:
            break

        # keep thresholding consistent with original (per-channel)
        ret, thresholded_frame = cv2.threshold(frame, 220, 255, cv2.THRESH_BINARY)

        # build a colored version of the binary image where white pixels map to a hue
        # that slowly rotates over the video. hue range in OpenCV is 0..179.
        # compute hue so it progresses across the whole video
        hue = int((frame_count / max(1, length - 1)) * 179)  # 0..179

        # obtain a single-channel mask (white pixels == 255)
        if thresholded_frame.ndim == 3:
            mask = cv2.cvtColor(thresholded_frame, cv2.COLOR_BGR2GRAY)
        else:
            mask = thresholded_frame

        # create a full-image color for this hue
        hsv = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = hue
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        color_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # map white pixels to the hue color, keep black pixels black
        color_frame = np.zeros_like(frame)
        white_mask = (mask == 255)
        # assign colored pixels where mask is white
        color_frame[white_mask] = color_bgr[white_mask]

        if frame_count == 0:
            # prev_frame = color_frame.copy()
            prev_frame = np.zeros_like(color_frame)
            # prev_frame_bw = thresholded_frame.copy()
            prev_frame_bw = np.zeros_like(thresholded_frame)


        # Add the new coloured frame onto prev_frame at 0.0025 opacity for the new frame
        # Alter the opacity to change how much influence the current frame has over the
        # previous ones. We have found 0.0025 to work well for most Zantics input videos
        # We are aiming to build up an average the highlights the trailing line but
        # does not show the '+'-shaped targets too strongly.

        opacity = 0.0025

        added_frame = cv2.addWeighted(prev_frame, 1.0, color_frame, opacity, 0)
        added_frame_bw = cv2.addWeighted(prev_frame_bw, 1.0, thresholded_frame, opacity, 0)

        prev_frame = added_frame
        prev_frame_bw = added_frame_bw
        frame_count += 1


        if SHOW:
            cv2.imshow("Threshold", thresholded_frame)
            cv2.imshow("Added", added_frame)

            # press 'q' or ESC to abort processing early
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("User requested exit.", file=sys.stderr)
                break

    # The final BW frame is white on a black background. For clarity and to enable
    # our coverage calculations, we invert it to be black on white.
    final_bw_frame = 255 - added_frame_bw

    if SHOW:
        cv2.imshow("Overlay", added_frame)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cap.release()

    # Get well coverages
    return get_well_coverages_from_final_frame(added_frame, final_bw_frame, out_dir, video_path.stem)




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

    # Loop through all videos in the "videos" folder, process them
    # and output to the 'results' folder. The coverage results will
    # be output to 'results/results.csv'.
    with open("frames/results.csv", "w", buffering=1) as results_file:
        results_file.write("video,well1_coverage,well2_coverage,well3_coverage,well4_coverage,well5_coverage,well6_coverage,well1_center_coverage,well2_center_coverage,well3_center_coverage,well4_center_coverage,well5_center_coverage,well6_center_coverage\n")
        results_file.flush()

        for path in sorted(videos_dir.iterdir()):
            if path.is_file() and path.suffix in VIDEO_EXTS:
                found = True
                out_dir = results_root / path.stem
                print(f"Processing {path.name} -> {out_dir}")

                well_coverages, well_center_coverages = split_and_stack_frames(path, out_dir)
                coverage_str = ",".join(f"{cov:.2f}" for cov in well_coverages)
                center_coverage_str = ",".join(f"{cov:.2f}" for cov in well_center_coverages)
                results_file.write(f"{path.name},{coverage_str},{center_coverage_str}\n")
                results_file.flush()

        if not found:
            print("No video files found in 'videos' folder.")

if __name__ == "__main__":
    main()