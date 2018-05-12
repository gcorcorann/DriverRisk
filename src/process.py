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
    # YOLO boxes
    objs_path = 'data/objects_new/' + vid_path[5:-3] + 'txt'
    # open video
    cap.open(vid_path)
    # open object boxes
    with open(objs_path, 'r') as f:
        objs = f.read().split('\n')[:-1]
    # for each frame in video

    for i in range(100):
        # frame objects
        frame_objs = objs[i].split()
        frame_objs = np.array(frame_objs, dtype=int).reshape(-1, 4)
        # read frame
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for obj in frame_objs:
            # left, top, right, bottom

            img_obj = frame[obj[1]:obj[3], obj[0]:obj[2]]
            save_path = 'data/processed/objects/' + vid_path[5:]
            print(save_path)
            break
            cv2.imshow('Object', img_obj)
            cv2.waitKey(0)

#    # convert to numpy array
#    X = np.array(X)
#    s = os.path.join(vid_path[:4], 'processed', vid_path[5:20])
#    # save to disk
#    np.save(s, X)

def main():
    """Main Function."""
    # create folder for processed videos
    os.system('rm -fr data/processed')
    os.system('mkdir data/processed')
    os.system('mkdir data/processed/objects')
    os.system('mkdir data/processed/objects/positive')
    os.system('mkdir data/processed/objects/negative')
    # read video paths
    pos_vids = glob.glob('data/positive/*.mp4')
    neg_vids = glob.glob('data/negative/*.mp4')
    all_vids = pos_vids + neg_vids
    cap = cv2.VideoCapture()

    # process each video
    for i, vid_path in enumerate(all_vids):
        # process video
        convert_to_numpy(cap, vid_path)
        break
        sys.stdout.write('\rCompletion: {:.0f}%'.format(i/len(all_vids)*100))

if __name__ == '__main__':
    main()

