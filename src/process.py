#!/usr/bin/env python3
import sys
import os
import glob
import cv2
import numpy as np

def convert_to_numpy(cap, vid_path):
    """Convert to ndarray and save to disk.
    Args:
        cap (cv2.VideoCapture): video capture object
        vid_path (string):      path to video.
    """
    # list to store processed frames
    X = []
    cap.open(vid_path)
    # for each frame in video
    for i in range(100):
        _, frame = cap.read()
        frame = cv2.resize(frame, (256,256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        X.append(frame)

    # convert to numpy array
    X = np.array(X)
    s = os.path.join(vid_path[:4], 'processed', vid_path[5:20])
    # save to disk
    np.save(s, X)

def main():
    """Main Function."""
    # create folder for processed videos
    os.system('rm -fr data/processed')
    os.system('mkdir data/processed')
    os.system('mkdir data/processed/positive')
    os.system('mkdir data/processed/negative')
    # read video paths
    pos_vids = glob.glob('data/positive/*.mp4')
    neg_vids = glob.glob('data/negative/*.mp4')
    all_vids = pos_vids + neg_vids
    cap = cv2.VideoCapture()

    # process each video
    for i, vid_path in enumerate(all_vids):
        # process video
        convert_to_numpy(cap, vid_path)
        sys.stdout.write('\rCompletion: {:.0f}%'.format(i/len(all_vids)*100))

if __name__ == '__main__':
    main()

