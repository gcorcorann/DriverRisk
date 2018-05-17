import sys
import cv2
import numpy as np

def draw_graph(w,h, label):
    graph = np.ones((h, 300, 3), dtype=np.uint8) * 255
    if label == 1:
        graph[3*h//4:, :] = (150, 215, 215)
    elif label == 2:
        graph[2*h//4:, :] = (0, 255, 255)
        graph[3*h//4:, :] = (150, 215, 215)
    elif label == 3:
        graph[h//4:, :] = (0, 100, 255)
        graph[2*h//4:, :] = (0, 255, 255)
        graph[3*h//4:, :] = (150, 215, 215)
    else:
        graph[:, :] = (0, 0, 255)
        graph[h//4:, :] = (0, 100, 255)
        graph[2*h//4:, :] = (0, 255, 255)
        graph[3*h//4:, :] = (150, 215, 215)


    cv2.line(graph, (0, h//4), (300,h//4), (0,0,0), 2)
    cv2.line(graph, (0, 2*h//4), (300,2*h//4), (0,0,0), 2)
    cv2.line(graph, (0, 3*h//4), (300,3*h//4), (0,0,0), 2)
    cv2.putText(graph, 'Low Risk', (75,7*h//8),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(graph, 'Moderate Risk', (40,5*h//8),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(graph, 'High Risk', (70,3*h//8),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(graph, 'Critical Risk', (55,1*h//8),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    return graph

def main():
    """Main Function."""
    fps = 20
    data_path = 'data/labels.txt'
    try:
        with open(data_path, 'r') as f:
            data = f.read().split()
            data = np.array(data).reshape(-1,2)
    except FileNotFoundError:
        print('Annotations file does not exist. Exiting program...')
        sys.exit()
    
    for i, instance in enumerate(data):
        vid_path, labels = instance
        print('Video Path: {}, Instance: {}/{}'.format(vid_path, i+1, 
            data.shape[0]))
        # convert to list of integers
        labels = [int(c) for c in labels]
        # open video
        cap = cv2.VideoCapture(vid_path)
        for i in range(100):
            _, frame = cap.read()
            h, w = frame.shape[:2]
            lab = labels[i]
            graph = draw_graph(w,h,lab)
            frame = np.hstack((frame, graph))
            cv2.imshow('Frame', frame)
            cv2.waitKey(int(1/fps*1000))

if __name__ == '__main__':
    main()

