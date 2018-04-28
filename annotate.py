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

class Annotate:
    """Class to annotate video clips."""

    cap = cv2.VideoCapture()
    #TODO remove comment
    #f = open('data/labels.txt', 'w')

    def __init__(self, fps):
        """Constructor.
        Args:
            fps (int):      fps of video stream
        """
        self.fps = fps
        self.video_paths = glob.glob('data/positive/*.mp4') \
                + glob.glob('data/negative/*.mp4')
        random.shuffle(self.video_paths)

    def __del__(self):
        """Deconstructor."""
        self.cap.release()
        #TODO remove comment
        #self.f.close()

    def print_message(self):
        """Print message with controls and annotations."""
        print('#' * 60)
        print('#' + ' '*58 + '#')
        print('#' + ' '*18 + 'DRIVER DANGER DETECTION' + ' '*17 + '#')
        print('#' + ' '*58 + '#')
        print('#' * 60)
        print('#' + ' '*58 + '#')
        print('# Low Risk: Ideal driving conditions where visual scenes   #')
        print('#\tdo not include any hazards.                        #')
        print('#' + ' '*58 + '#')
        print('# Medium Risk: Visual scenes that include potential        #')
        print('#\thazards with a low-to-medium probability to cause  #')
        print('#\tan incident.                                       #')
        print('#' + ' '*58 + '#')
        print('# High Risk: Visual scenes that include hazards with a     #')
        print('#\thigh probability to cause an incident              #')
        print('#\t(i.e. impending doom).                             #')
        print('#' + ' '*58 + '#')
        print('# Incident: Visual scences that depict an incident         #')
        print('#\t(e.g. accident).                                   #')
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
        print('#' + ' '*17 + 'Level 2 - Medium Risk' + ' '*20 + '#')
        print('#' + ' '*17 + 'Level 3 - High Risk' + ' '*22 + '#')
        print('#' + ' '*17 + 'Level 4 - Incident' + ' '*23 + '#')
        print('#' + ' '*58 + '#')
        print('#' * 60)

    def user_input(self):
        """Grabs user's input.

        Return:
            bool:       if true user quit program
        """
        # repeat until user's input is correct
        while True:
            label = input('Please enter input: ')
            if label == '1' or label == '2' or label == '3' or label == '4':
                #TODO write to file
                break

    def play_clip(self, frame_start):
        """Play segment of video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        for i in range(10):
            _, frame = self.cap.read()
            cv2.imshow('Frame', frame)
            #cv2.waitKey(int(1/self.fps*1000))
            cv2.waitKey(0)
        

    def play_video(self, vid_idx):
        """Play video."""
        self.cap.open(self.video_paths[vid_idx])
        # frame index
        f_idx = 0
        while f_idx is not 100:
            self.play_clip(f_idx)
            f_idx += 10
            print(f_idx)
            

    def run(self):
        """Run video annotator."""
        num_videos = len(self.video_paths)
        # current video index
        idx = 0
        while idx is not num_videos:
            print('Video index: {}/{}'.format(idx+1, num_videos))
            # store current index
            self.print_message()
            # open current video
            self.play_video(idx)
            # move to next video
            idx += 1
            break

def main():
    """Main Function."""
    #TODO change to 20
    fps = 20
    annotate = Annotate(fps)
    annotate.run()

if __name__ == '__main__':
    main()

