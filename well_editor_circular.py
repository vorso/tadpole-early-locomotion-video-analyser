#!/usr/bin/env python3
"""
Interactive Well Editor for Tadpole Locomotion Video Analysis - Circular Wells Version

This tool allows users to visually define and edit circular well positions for video analysis.
Features:
- Load video files and scrub through frames
- Load/save well configurations from/to CSV
- Drag well centers to reposition them
- Drag well edges to resize ALL wells simultaneously
- Add/remove wells
- Visual feedback with labeled circles
"""

from pathlib import Path
import cv2
import numpy as np
import csv
import argparse
import sys
import platform
from typing import List, Tuple, Optional
import copy
try:
    from tkinter import Tk, filedialog
    TKINTER_AVAILABLE = True
except:
    TKINTER_AVAILABLE = False

# Well data structure: [name, center_x, center_y, radius]
class Well:
    def __init__(self, name: str, center_x: int, center_y: int, radius: int):
        self.name = name
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.selected = False
        self.dragging = False
        self.resizing = False

    def get_bounds(self) -> Tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box for the circle"""
        return (
            self.center_x - self.radius,
            self.center_y - self.radius,
            self.center_x + self.radius,
            self.center_y + self.radius
        )

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside circular well"""
        distance = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        return distance <= self.radius

    def is_near_edge(self, x: int, y: int, threshold: int = 10) -> bool:
        """Check if point is near the circle edge for resizing"""
        distance = np.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        return abs(distance - self.radius) < threshold


class WellEditor:
    def __init__(self, video_path: str, wells_csv: Optional[str] = None):
        self.video_path = video_path
        self.wells_csv = wells_csv
        self.wells: List[Well] = []
        self.cap = None
        self.current_frame = None
        self.frame_number = 0
        self.total_frames = 0
        self.selected_well: Optional[Well] = None
        self.window_name = "Well Editor - Circular Wells"
        self.help_window_name = "Controls"
        self.help_visible = True
        self.playing = False
        self.play_fps = 30

        # Undo history (list of wells lists). Each entry is a deep-copied list of Well objects
        self.history = []
        self.history_limit = 50
        # Running flag to control main loop (used by Close button)
        self.running = True
        # Dirty flag: True if wells state changed since last save/open
        self.dirty = False
        # Internal flag to mark modifications during a drag/resize action
        self._modified_during_drag = False

        # Mouse state
        self.mouse_x = 0
        self.mouse_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.resizing_all = False
        self.resize_start_radius = 0
        self.resize_start_distance = 0

        # Load video
        self.load_video()

        # Load wells if CSV provided, otherwise auto-detect
        if wells_csv:
            self.load_wells_csv(wells_csv)
        else:
            # Try to auto-detect wells from first frame
            detected_wells = self.detect_wells_from_frame(self.current_frame)
            if detected_wells:
                self.wells = detected_wells
                print(f"Auto-detected {len(self.wells)} wells from first frame")
            else:
                # Fallback to default well if detection fails
                self.wells.append(Well("well1", 200, 200, 89))
                print("Auto-detection failed, created default well")

    def load_video(self):
        """Load video file"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            sys.exit(1)

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Loaded video: {self.video_path}")
        print(f"Total frames: {self.total_frames}")

        # Read first frame
        self.seek_frame(0)

    def seek_frame(self, frame_num: int):
        """Seek to specific frame"""
        frame_num = max(0, min(frame_num, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.frame_number = frame_num

    def detect_wells_from_frame(self, frame) -> List[Well]:
        """
        Automatically detect circular wells in the frame.
        Returns list of Well objects with uniform radius and no overlaps.
        """
        if frame is None:
            return []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            # Detect circles using HoughCircles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=80,   # Minimum distance between circle centers
                param1=50,    # Canny edge detection threshold
                param2=30,    # Accumulator threshold for circle detection
                minRadius=50, # Minimum circle radius
                maxRadius=150 # Maximum circle radius
            )

            if circles is None:
                print("No circles detected in frame")
                return []

            # Convert circles to wells
            circles = np.uint16(np.around(circles))
            circle_list = []

            for circle in circles[0, :]:
                center_x, center_y, radius = circle
                circle_list.append((int(center_x), int(center_y), int(radius)))

            # Calculate uniform radius based on largest detected circle plus minimal padding
            max_radius = max(r for _, _, r in circle_list)
            padding = 3  # Minimal padding
            uniform_radius = max_radius + padding

            print(f"Detected {len(circle_list)} circles")
            print(f"Using uniform well radius: {uniform_radius} pixels")

            # Create wells with uniform radius
            detected_wells = []
            for i, (center_x, center_y, radius) in enumerate(circle_list):
                well = Well(
                    f"well{i+1}",
                    center_x,
                    center_y,
                    uniform_radius
                )
                detected_wells.append(well)

            # Remove overlapping wells (keep the one with better position)
            detected_wells = self.remove_overlapping_wells(detected_wells)

            # Sort wells by position (top to bottom, left to right)
            detected_wells.sort(key=lambda w: (w.center_y // 100, w.center_x))

            # Renumber wells after sorting and filtering
            for i, well in enumerate(detected_wells):
                well.name = f"well{i+1}"

            print(f"Final well count after overlap removal: {len(detected_wells)}")

            return detected_wells

        except Exception as e:
            print(f"Error during well detection: {e}")
            return []

    def remove_overlapping_wells(self, wells: List[Well]) -> List[Well]:
        """Remove overlapping wells, keeping the better positioned ones"""
        if len(wells) <= 1:
            return wells

        non_overlapping = []

        for well in wells:
            overlaps = False
            for existing_well in non_overlapping:
                if self.wells_overlap(well, existing_well):
                    overlaps = True
                    print(f"Removing overlapping well at ({well.center_x}, {well.center_y})")
                    break

            if not overlaps:
                non_overlapping.append(well)

        return non_overlapping

    def wells_overlap(self, well1: Well, well2: Well) -> bool:
        """Check if two circular wells overlap"""
        distance = np.sqrt((well1.center_x - well2.center_x)**2 + (well1.center_y - well2.center_y)**2)
        return distance < (well1.radius + well2.radius)

    def load_wells_csv(self, csv_path: str):
        """Load wells from CSV file"""
        self.wells.clear()
        try:
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    well = Well(
                        row['well_name'],
                        int(row['center_x']),
                        int(row['center_y']),
                        int(row['radius'])
                    )
                    self.wells.append(well)
            print(f"Loaded {len(self.wells)} wells from {csv_path}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            self.wells.append(Well("well1", 200, 200, 89))

    def check_wells_uniform_radius(self) -> bool:
        """Check if all wells have the same radius"""
        if not self.wells:
            return True

        first_radius = self.wells[0].radius
        for well in self.wells[1:]:
            if well.radius != first_radius:
                return False
        return True

    def calculate_total_pixels(self, radius: int) -> int:
        """
        Calculate the total number of pixels in a circular well.
        Uses a mask to count exact pixels within the circle.
        """
        # Create a square mask with the circle
        img_size = radius * 2
        mask = np.zeros((img_size, img_size), dtype=np.uint8)

        # Draw filled circle
        center = (radius, radius)
        cv2.circle(mask, center, radius, 255, -1)

        # Count white pixels (pixels inside the circle)
        total_pixels = np.sum(mask == 255)

        return int(total_pixels)

    def save_wells_csv(self, csv_path: str = None):
        """Save wells to CSV file with full coverage image reference."""
        # Check if all wells have the same radius
        if not self.check_wells_uniform_radius():
            print("ERROR: Cannot save - all wells must have the same radius!")
            print("Drag any well edge to resize all wells to the same size.")
            return False
        # Save flow continues here (same function)
        if csv_path is None:
            # Set default filename
            default_name = "wells.csv"
            if self.wells_csv:
                default_name = Path(self.wells_csv).name

            # On macOS with tkinter issues, use console input
            if platform.system() == 'Darwin' or not TKINTER_AVAILABLE:
                print(f"\n=== Save Wells Configuration ===")
                print(f"Default filename: {default_name}")
                user_input = input(f"Enter filename (or press Enter for default): ").strip()
                if user_input:
                    csv_path = user_input
                    if not csv_path.endswith('.csv'):
                        csv_path += '.csv'
                else:
                    csv_path = default_name
                print(f"Saving to: {csv_path}")
            else:
                # Use file dialog on Windows/Linux
                try:
                    root = Tk()
                    root.withdraw()
                    csv_path = filedialog.asksaveasfilename(
                        title="Save Wells Configuration",
                        defaultextension=".csv",
                        initialfile=default_name,
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                    )
                    root.destroy()
                except Exception as e:
                    print(f"Error opening save dialog: {e}")
                    print("Falling back to console input")
                    user_input = input(f"Enter filename [{default_name}]: ").strip()
                    csv_path = user_input if user_input else default_name

                if not csv_path:  # User cancelled
                    print("Save cancelled")
                    return False

        try:
            # Calculate total pixels for the wells (all have same radius)
            if not self.wells:
                print("No wells to save")
                return False

            total_pixels = self.calculate_total_pixels(self.wells[0].radius)
            print(f"Calculated total pixels per well: {total_pixels}")

            # Save CSV with total_pixels column
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['well_name', 'center_x', 'center_y', 'radius', 'total_pixels'])
                for well in self.wells:
                    writer.writerow([well.name, well.center_x, well.center_y, well.radius, total_pixels])
            print(f"Saved {len(self.wells)} wells to {csv_path}")
            # Update the wells_csv path for future saves
            self.wells_csv = csv_path
            # mark as saved
            self.dirty = False
            return True
        except Exception as e:
            print(f"Error saving CSV: {e}")
            return False

    def open_wells_csv(self, csv_path: str = None):
        """Open wells CSV using file dialog or console input and load wells."""
        # Ask for filename if not provided
        if csv_path is None:
            if platform.system() == 'Darwin' or not TKINTER_AVAILABLE:
                user_input = input(f"Enter path to wells CSV to open: ").strip()
                if not user_input:
                    print("Open cancelled")
                    return False
                csv_path = user_input
            else:
                try:
                    root = Tk()
                    root.withdraw()
                    csv_path = filedialog.askopenfilename(
                        title="Open Wells Configuration",
                        defaultextension=".csv",
                        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                    )
                    root.destroy()
                except Exception as e:
                    print(f"Error opening file dialog: {e}")
                    user_input = input(f"Enter path to wells CSV to open: ").strip()
                    if not user_input:
                        print("Open cancelled")
                        return False
                    csv_path = user_input

            if not csv_path:
                print("Open cancelled")
                return False

        # Push current state so the user can undo the open
        self.push_history()
        try:
            self.load_wells_csv(csv_path)
            # remember path for future saves
            self.wells_csv = csv_path
            # Mark as not dirty since we loaded from file
            self.dirty = False
            print(f"Opened wells from: {csv_path}")
            return True
        except Exception as e:
            print(f"Failed to open wells CSV: {e}")
            return False

    def push_history(self):
        """Push current wells state onto history stack for undo."""
        # Deep copy wells list as new Well instances
        copied = [Well(well.name, int(well.center_x), int(well.center_y), int(well.radius)) for well in self.wells]
        # Preserve selection state
        for i, w in enumerate(self.wells):
            copied[i].selected = w.selected
        self.history.append(copied)
        # Enforce limit
        if len(self.history) > self.history_limit:
            self.history.pop(0)

    def undo(self):
        """Revert to the previous wells state if available."""
        if not self.history:
            print("Nothing to undo")
            return False
        last = self.history.pop()
        self.wells = last
        self.selected_well = None
        print("Undo: reverted to previous wells configuration")
        return True

    def add_well(self):
        """Add a new well"""
        # record state for undo
        self.push_history()
        # Find next available well number
        well_nums = []
        for well in self.wells:
            if well.name.startswith('well'):
                try:
                    num = int(well.name[4:])
                    well_nums.append(num)
                except:
                    pass

        next_num = max(well_nums) + 1 if well_nums else 1

        # Place new well at center of frame or offset from last well
        if self.wells:
            last_well = self.wells[-1]
            new_x = last_well.center_x + 50
            new_y = last_well.center_y
            new_radius = last_well.radius
        else:
            new_x = self.current_frame.shape[1] // 2
            new_y = self.current_frame.shape[0] // 2
            new_radius = 89

        new_well = Well(f"well{next_num}", new_x, new_y, new_radius)
        self.wells.append(new_well)
        # mark as dirty since we've changed the wells
        self.dirty = True
        print(f"Added {new_well.name}")

    def remove_selected_well(self):
        """Remove currently selected well"""
        if self.selected_well and self.selected_well in self.wells:
            # record state for undo
            self.push_history()
            name = self.selected_well.name
            self.wells.remove(self.selected_well)
            self.selected_well = None
            # mark as dirty since we've changed the wells
            self.dirty = True
            print(f"Removed {name}")

    def draw_wells(self, frame):
        """Draw circular wells on frame"""
        display = frame.copy()

        for well in self.wells:
            # Choose color based on selection
            if well.selected or well == self.selected_well:
                color = (0, 255, 0)  # Green for selected
                thickness = 3
            else:
                color = (0, 165, 255)  # Orange for normal
                thickness = 2

            # Draw circle
            cv2.circle(display, (well.center_x, well.center_y), well.radius, color, thickness)

            # Draw center point
            cv2.circle(display, (well.center_x, well.center_y), 5, color, -1)

            # Draw edge indicator for selected well
            if well.selected or well == self.selected_well:
                # Draw small circles at cardinal points on the edge
                for angle in [0, 90, 180, 270]:
                    rad = np.radians(angle)
                    edge_x = int(well.center_x + well.radius * np.cos(rad))
                    edge_y = int(well.center_y + well.radius * np.sin(rad))
                    cv2.circle(display, (edge_x, edge_y), 6, (255, 0, 0), -1)

            # Draw label
            label = f"{well.name} (r={well.radius})"
            label_pos = (well.center_x - well.radius, well.center_y - well.radius - 10)
            if label_pos[1] < 30:
                label_pos = (well.center_x - well.radius, well.center_y + well.radius + 25)
            # Draw label with a white halo outline so black text remains readable
            # Draw thicker white text first (halo), then black text on top
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            # white halo
            cv2.putText(display, label, label_pos, font, font_scale, (255, 255, 255), 3, cv2.LINE_AA)
            # main black text
            cv2.putText(display, label, label_pos, font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

        return display

    def draw_buttons(self, display):
        """Draw UI buttons (Undo, Save, Close) and store their hit rectangles."""
        h, w = display.shape[:2]
        button_w = 120
        button_h = 30
        margin = 10
        # right-aligned buttons stacked vertically
        x1 = w - margin - button_w
        y = margin

        # Define buttons and store rects
        self.button_rects = {}
        for name in ["Undo", "Save", "Open", "Close"]:
            x2 = x1 + button_w
            y2 = y + button_h
            # Button background
            # highlight when mouse over
            if hasattr(self, 'mouse_x') and hasattr(self, 'mouse_y') and (x1 <= self.mouse_x <= x2 and y <= self.mouse_y <= y2):
                bg = (50, 200, 50)
                fg = (0, 0, 0)
            else:
                bg = (200, 200, 200)
                fg = (0, 0, 0)

            cv2.rectangle(display, (x1, y), (x2, y2), bg, -1)
            cv2.rectangle(display, (x1, y), (x2, y2), (0, 0, 0), 1)
            txt_sz = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            txt_x = x1 + (button_w - txt_sz[0]) // 2
            txt_y = y + (button_h + txt_sz[1]) // 2
            cv2.putText(display, name, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)
            self.button_rects[name.lower()] = (x1, y, x2, y2)

            y += button_h + 8
        return display

    def draw_current_wells_label(self, display):
        """Draw the currently opened/saved wells CSV filename on the display."""
        h, w = display.shape[:2]
        # Use basename of wells_csv if available
        if self.wells_csv:
            try:
                name = Path(self.wells_csv).name
            except Exception:
                name = str(self.wells_csv)
        else:
            name = "(no wells file)"

        label = f"Wells: {name}"
        if self.dirty:
            label += " *"  # unsaved marker
        txt_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        pad = 6
        rect_w = txt_sz[0] + pad * 2
        rect_h = txt_sz[1] + pad * 2

        x = 10
        y = 10

        # Background rectangle (semi-transparent effect by solid fill)
        cv2.rectangle(display, (x, y), (x + rect_w, y + rect_h), (50, 50, 50), -1)
        # Text baseline position
        txt_x = x + pad
        txt_y = y + pad + txt_sz[1]
        cv2.putText(display, label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return display

    def _shorten_middle(self, s: str, max_len: int) -> str:
        """Shorten a string in the middle with ellipsis if it's longer than max_len.

        Keeps the start and end of the string and inserts '...' in the middle.
        """
        if s is None:
            return ""
        s = str(s)
        if len(s) <= max_len:
            return s
        if max_len <= 6:
            return s[:max_len]
        # allocate space for head and tail around the '...'
        keep = max_len - 3
        head = keep // 2
        tail = keep - head
        return s[:head] + '...' + s[-tail:]

    def draw_status_bar(self, display):
        """Draw a small status bar at the bottom showing filename and undo depth."""
        h, w = display.shape[:2]
        # Show full path in status bar (truncate if too long to fit)
        if self.wells_csv:
            try:
                fullpath = str(Path(self.wells_csv))
            except Exception:
                fullpath = str(self.wells_csv)
        else:
            fullpath = "(no wells file)"

        undo_count = len(self.history)
        dirty_marker = " *" if self.dirty else ""

        # Build label using full path first
        label = f"File: {fullpath}{dirty_marker}   |   Undo: {undo_count}"

        # Measure and shorten path if the rendered text would be wider than the window
        txt_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        max_width = w - 20
        if txt_sz[0] > max_width:
            # Try a couple of shortening passes with decreasing max lengths until it fits
            for target_chars in (120, 80, 60, 40, 30):
                short_path = self._shorten_middle(fullpath, target_chars)
                label = f"File: {short_path}{dirty_marker}   |   Undo: {undo_count}"
                txt_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                if txt_sz[0] <= max_width:
                    break

        pad = 8
        rect_h = txt_sz[1] + pad * 2
        y = h - rect_h - 6
        # Background
        cv2.rectangle(display, (0, y), (w, y + rect_h), (30, 30, 30), -1)
        # Text
        txt_x = 10
        txt_y = y + pad + txt_sz[1]
        cv2.putText(display, label, (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        return display

    def create_help_panel(self):
        """Create a separate help panel window"""
        help_text = [
            ("CONTROLS", (255, 255, 0)),
            ("", (255, 255, 255)),
            ("Video Playback:", (200, 200, 255)),
            ("  SPACE - Play/Pause", (255, 255, 255)),
            ("  Left/Right - Frame step", (255, 255, 255)),
            ("  Trackbar - Scrub frames", (255, 255, 255)),
            ("", (255, 255, 255)),
            ("Well Editing:", (200, 200, 255)),
            ("  Left-click center - Select/move well", (255, 255, 255)),
            ("  Drag edge - Resize ALL wells", (255, 255, 255)),
            ("  U - Undo", (255, 255, 255)),
            ("  A - Add new well", (255, 255, 255)),
            ("  D - Delete selected well", (255, 255, 255)),
            ("", (255, 255, 255)),
            ("File Operations:", (200, 200, 255)),
            ("  W - Save (opens dialog)", (255, 255, 255)),
            ("  O - Open (load wells)", (255, 255, 255)),
            ("", (255, 255, 255)),
            ("Window:", (200, 200, 255)),
            ("  H - Toggle this help panel", (255, 255, 255)),
            ("  Q/ESC - Quit editor", (255, 255, 255)),
        ]

        # Create help panel image
        panel_width = 350
        line_height = 25
        panel_height = len(help_text) * line_height + 40
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background

        y_offset = 30
        for i, (line, color) in enumerate(help_text):
            cv2.putText(panel, line, (10, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return panel

    def show_help_panel(self):
        """Show or update the help panel window"""
        if self.help_visible:
            panel = self.create_help_panel()
            cv2.imshow(self.help_window_name, panel)
        else:
            cv2.destroyWindow(self.help_window_name)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        self.mouse_x = x
        self.mouse_y = y

        # If user clicked a UI button, handle that and return
        if event == cv2.EVENT_LBUTTONDOWN and hasattr(self, 'button_rects'):
            for name, rect in self.button_rects.items():
                x1, y1, x2, y2 = rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if name == 'undo':
                        self.undo()
                    elif name == 'save':
                        if self.save_wells_csv():
                            print("Save successful!")
                    elif name == 'open':
                        if self.open_wells_csv():
                            print("Open successful")
                    elif name == 'close':
                        print("Close requested via button")
                        self.running = False
                    return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a well
            clicked_well = None
            for well in reversed(self.wells):  # Check from top to bottom
                # First check if clicking near edge for resizing ALL wells
                if well.is_near_edge(x, y):
                    # record state for undo before starting resize
                    self.push_history()
                    self.resizing_all = True
                    self.resize_start_distance = np.sqrt((x - well.center_x)**2 + (y - well.center_y)**2)
                    self.resize_start_radius = well.radius
                    well.selected = True
                    self.selected_well = well
                    clicked_well = well
                    print("Resizing all wells...")
                    break
                # Then check if clicking inside well center
                elif well.contains_point(x, y):
                    # record state for undo before starting drag
                    self.push_history()
                    well.dragging = True
                    well.selected = True
                    self.selected_well = well
                    self.drag_start_x = x - well.center_x
                    self.drag_start_y = y - well.center_y
                    clicked_well = well
                    break

            # Deselect other wells
            for well in self.wells:
                if well != clicked_well:
                    well.selected = False

        elif event == cv2.EVENT_MOUSEMOVE:
            # Handle dragging or resizing
            if self.resizing_all:
                # Resize ALL wells based on distance change
                for well in self.wells:
                    if well == self.selected_well:
                        current_distance = np.sqrt((x - well.center_x)**2 + (y - well.center_y)**2)
                        radius_change = int(current_distance - self.resize_start_distance)
                        new_radius = max(20, self.resize_start_radius + radius_change)

                        # Apply same radius to ALL wells
                        for w in self.wells:
                            w.radius = new_radius
                        # Mark that we modified during this drag/resize so LBUTTONUP can mark dirty
                        self._modified_during_drag = True
                        break
            else:
                # Handle individual well dragging
                for well in self.wells:
                    if well.dragging:
                        well.center_x = x - self.drag_start_x
                        well.center_y = y - self.drag_start_y
                        # Mark modification
                        self._modified_during_drag = True

        elif event == cv2.EVENT_LBUTTONUP:
            # Stop dragging/resizing
            self.resizing_all = False
            for well in self.wells:
                well.dragging = False
                well.resizing = False
            # If any modification happened during the drag, mark as dirty
            if self._modified_during_drag:
                self.dirty = True
                self._modified_during_drag = False

    def frame_trackbar_callback(self, value):
        """Handle frame trackbar changes - updates immediately during drag"""
        self.seek_frame(value)
        # Force immediate display update during scrubbing
        if self.current_frame is not None:
            display = self.draw_wells(self.current_frame)
            info_text = f"Frame: {self.frame_number}/{self.total_frames-1} | Wells: {len(self.wells)} | SCRUBBING"
            cv2.putText(display, info_text, (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, info_text, (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.imshow(self.window_name, display)

    def run(self):
        """Main editor loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Create trackbar for frame selection
        cv2.createTrackbar('Frame', self.window_name, 0, self.total_frames - 1,
                          self.frame_trackbar_callback)

        # Show help panel initially
        self.show_help_panel()

        print("\n=== Well Editor Started (Circular Wells) ===")
        print("Press 'H' to toggle help panel")
        print("Press SPACE to play/pause video")
        print("Drag well centers to move, drag edges to resize ALL wells")

        # Calculate delay for playback
        delay_ms = int(1000 / self.play_fps)

        while self.running:
            if self.current_frame is None:
                break

            # Handle video playback
            if self.playing:
                next_frame = self.frame_number + 1
                if next_frame >= self.total_frames:
                    next_frame = 0  # Loop back to start
                self.seek_frame(next_frame)
                # Update trackbar during playback
                cv2.setTrackbarPos('Frame', self.window_name, self.frame_number)

            # Draw wells on current frame
            display = self.draw_wells(self.current_frame)
            # Draw UI buttons
            display = self.draw_buttons(display)
            # Draw current wells filename label
            display = self.draw_current_wells_label(display)
            # Draw status bar (filename + undo depth)
            display = self.draw_status_bar(display)

            # Show frame info
            play_status = "PLAYING" if self.playing else "PAUSED"
            info_text = f"Frame: {self.frame_number}/{self.total_frames-1} | Wells: {len(self.wells)} | {play_status}"
            cv2.putText(display, info_text, (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, info_text, (10, display.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            cv2.imshow(self.window_name, display)

            # Handle keyboard input with appropriate delay
            wait_time = delay_ms if self.playing else 1  # Very short delay for responsive scrubbing
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                self.running = False
                break
            elif key == ord(' '):  # Space - Play/Pause
                self.playing = not self.playing
                print(f"Video {'playing' if self.playing else 'paused'}")
            elif key == ord('h'):  # Toggle help
                self.help_visible = not self.help_visible
                self.show_help_panel()
            elif key == ord('a'):  # Add well
                self.add_well()
            elif key == ord('d'):  # Delete selected well
                self.remove_selected_well()
            elif key == ord('w'):  # Save with dialog
                if self.save_wells_csv():
                    print("Save successful!")
            elif key == ord('u'):
                # Undo
                self.undo()
            elif key == ord('s'):
                # quick save (calls same save dialog)
                if self.save_wells_csv():
                    print("Save successful!")
            elif key == ord('o'):
                # Open wells CSV
                if self.open_wells_csv():
                    print("Open successful")
            elif key == ord('c'):
                # close editor
                print("Close requested via key")
                self.running = False
            elif key == 81 or key == 2:  # Left arrow
                self.playing = False
                self.seek_frame(self.frame_number - 1)
            elif key == 83 or key == 3:  # Right arrow
                self.playing = False
                self.seek_frame(self.frame_number + 1)

        cv2.destroyAllWindows()
        self.cap.release()
        print("\n=== Well Editor Closed ===")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Well Editor for Tadpole Locomotion Video Analysis (Circular Wells)'
    )
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('--wells-csv', type=str, default=None,
                       help='Path to wells CSV file to load/save (optional)')

    args = parser.parse_args()

    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Create and run editor
    editor = WellEditor(args.video, args.wells_csv)
    editor.run()


if __name__ == "__main__":
    main()
