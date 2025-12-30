#Install prerequisites

sudo apt update && sudo apt install python3-pip -y;
sudo pip install opencv-python pandas;

#Run Tracker

python3 xen_loco_frame_stacker
python3 create_coverage_composite.py

