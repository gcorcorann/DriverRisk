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
    obj_dir = 'data/processed/flow/objects/' + vid_path[12:-4]
    os.system('mkdir ' + obj_dir)
    # open video
    cap = cv2.VideoCapture(vid_path)
    # objects_path
    objs_path = vid_path[:5] + 'objects/' + vid_path[12:-3] + 'txt'
    # read objeect bounding boxes
    with open(objs_path, 'r') as f:
        objs = f.read().split('\n')[:-1]

    # list to hold flow images
    X_flow = []
    # read initial frame
    _, frame1 = cap.read()
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # create flow image
    flow_img = np.zeros_like(frame1)
    # for each frame in video
    for i in range(1, 100):
        # frame objects
        frame_objs = objs[i].split()
        frame_objs = np.array(frame_objs, dtype=int).reshape(-1, 4)
        _, frame2 = cap.read()
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # compute flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15,
                3, 5, 1.2, 0)
        # compute flow magnitude
        mag, _ = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
        # mean subtraction
        flow[:, :, 0] -= np.mean(flow[:,:,0])
        flow[:, :, 1] -= np.mean(flow[:,:,1])
        mag -= np.mean(mag)
        # normalize
        flow[:, :, 0] = cv2.normalize(flow[:,:,0], None, 0, 255,
                cv2.NORM_MINMAX)
        flow[:, :, 1] = cv2.normalize(flow[:,:,1], None, 0, 255,
                cv2.NORM_MINMAX)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_img[:, :, 0] = np.uint8(flow[:, :, 0])
        flow_img[:, :, 1] = np.uint8(flow[:, :, 1])
        flow_img[:, :, 2] = np.uint8(mag)

        # extrack objects in frame
        extract_objects(obj_dir, flow_img, i, frame_objs)

        # store copy
        X_flow.append(np.copy(flow_img))
        # set current image to previous
        gray1 = gray2

    X_flow = np.array(X_flow)
    save_path = 'data/processed/flow/videos/' + vid_path[12:-3] + 'npy'
    np.save(save_path, X_flow)

def make_directories():
    # create folders for processed objects and videos
    os.system('rm -fr data/processed/flow')
    os.system('mkdir data/processed/flow')
    os.system('mkdir data/processed/flow/objects')
    os.system('mkdir data/processed/flow/videos')
    os.system('mkdir data/processed/flow/videos/positive')
    os.system('mkdir data/processed/flow/videos/negative')
    os.system('mkdir data/processed/flow/objects/positive')
    os.system('mkdir data/processed/flow/objects/negative')

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
        print('Iteration: {}, Video Path: {}'.format(i+1, vid_path))
        process_video(vid_path)

if __name__ == '__main__':
    main()

