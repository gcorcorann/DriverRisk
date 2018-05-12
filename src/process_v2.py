#!/usr/bin/env python3
import glob
import cv2
import numpy as np
import os

def extract_objects(obj_dir, frame, frame_idx, objs):
    for i, obj in enumerate(objs):
        # left, top, right, bottom
        left, top, right, bottom = obj
        img_obj = frame[top:bottom, left:right]
        save_path = obj_dir + '/' + str(frame_idx) + '-' + str(i) + '.png'
        cv2.imwrite(save_path, img_obj)

def process_video(vid_path):
    # make new directory for all objects in this video
    obj_dir = 'data/processed/objects/' + vid_path[5:-4]
    os.system('mkdir ' + obj_dir)
    # open video
    cap = cv2.VideoCapture(vid_path)
    # objects path
    objs_path = vid_path[:5] + 'objects_new/' + vid_path[5:-3] + 'txt'
    # read object bounding boxes
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

        # extract objects in frame
        extract_objects(obj_dir, frame, i, frame_objs)

def make_directories():
    # create folders for processed videos
    os.system('rm -fr data/processed')
    os.system('mkdir data/processed')
    os.system('mkdir data/processed/positive')
    os.system('mkdir data/processed/negative')
    os.system('mkdir data/processed/objects')
    os.system('mkdir data/processed/objects/positive')
    os.system('mkdir data/processed/objects/negative')

def main():
    """Main Function."""
    # read video paths
    pos_vids = glob.glob('data/positive/*.mp4')
    neg_vids = glob.glob('data/negative/*.mp4')
    all_vids = pos_vids + neg_vids
    # make processed data directories
    make_directories()

    # process video
    for i, vid_path in enumerate(all_vids):
        print('i =', i)
        process_video(vid_path)

if __name__ == '__main__':
    main()

