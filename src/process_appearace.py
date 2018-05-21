#!/usr/bin/env python3
import random
import glob
import cv2
import numpy as np
import os

def extract_objects(obj_dir, frame, frame_idx, objs):
    for i, obj in enumerate(objs):
        # left, top, right, bottom
        left, top, right, bottom = obj
        img_obj = frame[top:bottom, left:right]
        save_path = obj_dir + '/{:02}'.format(frame_idx) + '-{:02}'.format(i) \
                + '.png'
        cv2.imwrite(save_path, img_obj)

def process_video(vid_path):
    # make new directory for all objects in this video
    obj_dir = 'data/processed/appearance/objects/' + vid_path[12:-4]
    os.system('mkdir ' + obj_dir)
    # open video
    cap = cv2.VideoCapture(vid_path)
    # objects path
    objs_path = vid_path[:5] + 'objects/' + vid_path[12:-3] + 'txt'
    # read object bounding boxes
    with open(objs_path, 'r') as f:
        objs = f.read().split('\n')[:-1]

    # for each frame in video
    X = []
    for i in range(100):
        # frame objects
        frame_objs = objs[i].split()
        frame_objs = np.array(frame_objs, dtype=int).reshape(-1, 4)
        # read frame
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # extract objects in frame
        extract_objects(obj_dir, frame, i, frame_objs)

        frame = cv2.resize(frame, (256, 256))
        X.append(frame)

    X = np.array(X)
    save_path = 'data/processed/appearance/videos/' + vid_path[12:-3] + 'npy'
    np.save(save_path, X)

def make_directories():
    # create folders for processed objects and videos
    os.system('rm -fr data/processed/appearance')
    os.system('mkdir data/processed/appearance')
    os.system('mkdir data/processed/appearance/objects')
    os.system('mkdir data/processed/appearance/videos')
    os.system('mkdir data/processed/appearance/videos/positive')
    os.system('mkdir data/processed/appearance/videos/negative')
    os.system('mkdir data/processed/appearance/objects/positive')
    os.system('mkdir data/processed/appearance/objects/negative')

def main():
    """Main Function."""
    # read video paths
    video_path = 'data/videos/'
    pos_vids = glob.glob(video_path + 'positive/*.mp4')
    neg_vids = glob.glob(video_path + 'negative/*.mp4')
    all_vids = pos_vids + neg_vids
    random.shuffle(all_vids)
    # make processed data directories
    make_directories()

    # process each video
    for i, vid_path in enumerate(all_vids):
        print('i =', i)
        process_video(vid_path)

if __name__ == '__main__':
    main()

