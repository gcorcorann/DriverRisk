#!/usr/bin/env python3
import glob
import random

def main():
    """Main Function."""
    pos_objects = glob.glob('data/objects_new/positive/*.txt')
    neg_objects = glob.glob('data/objects_new/negative/*.txt')
    all_objects = pos_objects + neg_objects
    random.shuffle(all_objects)

    for i, obj_path in enumerate(all_objects):
        print('i = {}, object path = {}'.format(i+1, obj_path))
        with open(obj_path, 'r') as f:
            data = f.read()
            print(data)

        break

if __name__ == '__main__':
    main()

