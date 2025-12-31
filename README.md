# tadpole-locomotion-video-analyser
Simple script for post-analysing Xenopus tadpole locomotion videos without XY data. Often early locomotion tracker software will render videos with crosshairs integrated into the video without producing a clean video to reanalyse. This python script thresholds and overlays each frame of the crosshair video to produce a trail using hte position of the target in the video.

# Prerequisites

Requires Python3 and OpenCV
