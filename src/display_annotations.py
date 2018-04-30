import sys
import cv2
import numpy as np

def interpolate(p1, p2):
    m = (p2-p1)/10
    interp = [p1 + i * m for i in range(1,11)]
    return np.array(interp)

def create_labels(labels):
    """Interpolate missing labels.
    Args:
        labels (list):      labels every 10 frames apart
    
    Returns:
        list:               interpolated labels for 100 frames
    """
    labels_interp = np.zeros(len(labels)*10)
    labels_interp[:10] = labels[0]
    for i in range(len(labels)-1):
        labels_interp[(i+1)*10:(i+1)*10+10] = interpolate(labels[i], 
                labels[i+1])
    
    # remove floating point percision errors with rounding
    labels_interp = np.round(labels_interp+0.01)
    labels_interp = labels_interp.astype(int)
    return labels_interp

def draw_graph(w,h, label):
    graph = np.ones((h, 300, 3), dtype=np.uint8) * 255
    graph[(4-label)*h//4:, :] = (30,30,150)
    cv2.line(graph, (0, h//4), (300,h//4), (0,0,0), 2)
    cv2.line(graph, (0, 2*h//4), (300,2*h//4), (0,0,0), 2)
    cv2.line(graph, (0, 3*h//4), (300,3*h//4), (0,0,0), 2)
    cv2.putText(graph, 'Low Risk', (75,7*h//8),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(graph, 'Medium Risk', (45,5*h//8),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(graph, 'High Risk', (70,3*h//8),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(graph, 'Incident', (85,1*h//8),
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
        # create dense labels (interpolate missing values)
        labels = create_labels(labels)
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

