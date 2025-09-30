# vorso 2025

from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

cross_template = cv.imread('cross_template.png')

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='Xen_loco-20250507T101557-f30.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

vid_width  = capture.get(3)
vid_height = capture.get(4)

frame_count = 0;

length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

while frame_count < length:
    ret, frame = capture.read()

    print(f"Processing frame number {frame_count} of {args.input}")

    if frame is None:
        break

    ret,thresholded_frame = cv.threshold(frame,220,255,cv.THRESH_BINARY)

    if frame_count == 0:
        prev_frame = thresholded_frame

    added_frame = cv.add(thresholded_frame,prev_frame)

    prev_frame = added_frame
    frame_count += 1;

# Write the inverted image (255 - pixel value to invert, you can remove this if you want non-inverted)
cv.imwrite(f'{args.input}-overlay.png',255-added_frame)
