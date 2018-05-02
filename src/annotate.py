#!/usr/bin/env python3
"""Annotate each video clip.

For each 100 frame video clip, provide a danger level every 10 frames.
The missing values are then interpolated to provide a dense annotation.

USAGE:  python src/annotate.py
"""
import glob
import random
import cv2
import numpy as np
import time
import os.path
import sys

class Annotate:
    """Class to annotate video clips."""

    cap = cv2.VideoCapture()
    # index of video to annotate
    vid_idx = 0
    # index of frame to annotate
    frame_idx = 0
    # current video annotations
    annotations = []

    def __init__(self, fps):
        """Constructor.
        Args:
            fps (int):      fps of video stream
        """
        labels_path = 'data/labels.txt'
        # check if labels path already exists
        if os.path.isfile(labels_path):
            try:
                while True:
                    inp = input('Labels path already extis, do you want to ' \
                            + 'overwrite? [y/n]: ')
                    if inp is 'n':
                        raise NameError('Quitting Program')
                    elif inp is 'y':
                        break
            except NameError:
                print('Cannot overwrite file. Exiting program...')
                sys.exit()
            
        # if no annotations or exceptions
        self.f = open(labels_path, 'w')
        self.fps = fps
        self.video_paths = glob.glob('data/positive/*.mp4') \
                + glob.glob('data/negative/*.mp4')
        random.shuffle(self.video_paths)

    def __del__(self):
        """Deconstructor."""
        self.cap.release()
        try:
            self.f.close()
        except AttributeError:
            pass

    def print_message(self):
        """Print message with controls and annotation labels."""
        print('#' * 60)
        print('#' + ' '*58 + '#')
        print('#' + ' '*18 + 'DRIVER DANGER DETECTION' + ' '*17 + '#')
        print('#' + ' '*58 + '#')
        print('#' * 60)
        print('#' + ' '*58 + '#')
        print('# Low Risk: Visual scenes that do not include any          #')
        print('#\thazards (i.e. ideal driving setting).              #')
        print('#' + ' '*58 + '#')
        print('# Moderate Risk: Visual scenes that include potential      #')
        print('#\thazards with a low-to-medium probability to cause  #')
        print('#\tan incident (i.e. normal driving setting).         #')
        print('#' + ' '*58 + '#')
        print('# High Risk: Visual scenes that include hazards with a     #')
        print('#\thigh probability to cause an incident              #')
        print('#\t(i.e. unsafe driving setting).                     #')
        print('#' + ' '*58 + '#')
        print('# Critical Risk: Visual scences that depict a sure         #')
        print('#\tincident (i.e. impending doom).                    #')
        print('#' + ' '*58 + '#')
        print('#' * 60)
        print('#' + ' '*58 + '#')
        print('# Controls:' + ' '*48 + '#')
        print("#\t Input 'r' to replay current video segment." + ' '*8 + '#')
        print("#\t Input 'p' to play previous video segment." + ' '*9 + '#')
        print("#\t Input 'pp' to play previous video." + ' '*16 + '#')
        print('#' + ' '*58 + '#')
        print('#' + ' '*18 + 'Video Annotation Table' + ' '*18 + '#')
        print('#' + ' '*17 + 'Level 1 - Low Risk' + ' '*23 + '#')
        print('#' + ' '*17 + 'Level 2 - Moderate Risk' + ' '*18 + '#')
        print('#' + ' '*17 + 'Level 3 - High Risk' + ' '*22 + '#')
        print('#' + ' '*17 + 'Level 4 - Critical Risk' + ' '*18 + '#')
        print('#' + ' '*58 + '#')
        print('#' * 60)

    def __delete_annotation(self):
        """Delete last line in annotations."""
        # close write file
        self.f.close()
        # open read file
        f_read = open('data/labels.txt', 'r')
        lines = f_read.readlines()
        # close read file
        f_read.close()
        # open write file
        self.f = open('data/labels.txt', 'w')
        # remove last line
        for line in lines[:-1]:
            self.f.write(line)

    def user_input(self):
        """Grabs user's input.
        
        Return:
            bool:   True if we moving to previous video, else False  
        """
        try:
            # repeat until user's input is correct
            while True:
                label = input('Please enter input: ')
                if label == '1' or label == '2' or label == '3' \
                        or label == '4':
                    # store in annotations list
                    self.annotations.append(label)
                    return False
                elif label == 'pp':
                    # play previous video
                    if self.vid_idx - 1 < 0:
                        print('No previous video.')
                    else:
                        # remove 2 since we add 1 in run()
                        self.vid_idx -= 2
                        self.__delete_annotation()
                        return True
                elif label == 'r':
                    # replay video
                    self.frame_idx -= 10
                    return False
                elif label == 'p':
                    # play previous video segment
                    if self.frame_idx - 10 < 0:
                        print('No previous video segment.')
                    else:
                        # remove 20 since we add 10 in annotate_video()
                        self.frame_idx -= 20
                        # pop off last annotation in list
                        self.annotations.pop()
                        return False
                elif label == 'q':
                    raise NameError('Quit Program')
    
                print('Incorrect input.')
        except NameError:
            print('User quit. Exiting program...')
            sys.exit()

    def play_clip(self):
        """Play 10 frame segment of video."""
        # set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        for i in range(10):
            _, frame = self.cap.read()
            cv2.imshow('Frame', frame)
            cv2.waitKey(int(1/self.fps*1000))
        
    def annotate_video(self):
        """Annotate video every 10 frames."""
        # open video at current video index
        vid_path = self.video_paths[self.vid_idx]
        self.cap.open(vid_path)
        # reset frame idx
        self.frame_idx = 0
        # reset annotations
        self.annotations = []
        # step through video every 10 frames
        while self.frame_idx is not 100:
            self.play_clip()
            if self.user_input():
                break

            # go to next 10 frame segment
            self.frame_idx += 10

        if self.frame_idx == 100:
            # save labels
            vid_path = self.video_paths[self.vid_idx]
            # interpolate missing values
            labels = self.__fill_missing()
            self.f.write(vid_path + ' ' + ''.join(labels) + '\n')
    
    def __interpolate(self, p1, p2):
        m = (p2 - p1) / 10
        interp = [p1 + i * m for i in range(10)]
        return np.array(interp)

    def __fill_missing(self):
        labels = self.annotations
        labels = [int(c) for c in labels]
        print(labels)
        labels_interp = np.zeros(len(labels)*10)
        labels_interp[:10] = labels[0]
        for i in range(len(labels)-1):
            labels_interp[(i+1)*10: (i+2)*10] = self.__interpolate(labels[i],
                    labels[i+1])

        # remove floating point precision errors with rounding
        labels_interp = np.round(labels_interp+0.01)
        labels_interp = labels_interp.astype(int)
        labels_interp = [str(c) for c in labels_interp]
        return labels_interp

    def run(self):
        """Run video annotator."""
        num_videos = len(self.video_paths)
        # current video index
        while self.vid_idx is not num_videos:
            print('Video index: {}/{}'.format(self.vid_idx+1, num_videos))
            self.print_message()
            # annotate current video
            self.annotate_video()
            # move to next video
            self.vid_idx += 1

def main():
    """Main Function."""
    fps = 20
    annotate = Annotate(fps)
    annotate.run()

if __name__ == '__main__':
    main()

