#Install prerequisites

sudo apt update && sudo apt install python3-pip -y;
sudo pip install opencv-python pandas;

#Run Tracker
python3 tadpole_tracker.py --input my_video_file.avi;
