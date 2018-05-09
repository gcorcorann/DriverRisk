#!/usr/bin/env python3
import glob
import random

def main():
    """Main Function."""
    pos_objects = glob.glob('data/objects/positive/*.txt')
    neg_objects = glob.glob('data/objects/negative/*.txt')
    all_objects = pos_objects + neg_objects
    #random.shuffle(all_objects)

    for i, obj_path in enumerate(all_objects):
        print('i =', i+1)
        obj_dict = {}
        with open(obj_path, 'r') as f:
            for line in f:
                if line[:5] == 'Frame':
                    frame_num = line[6:-1]
                    obj_dict[frame_num] = []
                elif line[:5] == 'Bound':
                    # left, top, right, bottom
                    box = line.split()[2:]
                    box[0] = int(box[0][5:-1])
                    box[1] = int(box[1][4:-1])
                    box[2] = int(box[2][6:-1])
                    box[3] = int(box[3][7:])
                    obj_dict[frame_num].append(box)

        obj_path_new = obj_path[:12] + '_new' + obj_path[12:]
        with open(obj_path_new, 'w') as f:
            for key in obj_dict:
                if len(obj_dict[key]) > 20:
                    obj_dict[key] = obj_dict[key][:20]

                s = str(obj_dict[key])
                s = s.replace('[', '')
                s = s.replace(']', '')
                s = s.replace(',', '')
                f.write(s + '\n')

if __name__ == '__main__':
    main()

