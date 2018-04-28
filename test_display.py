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

def main():
    """Main Function."""
    fps = 20
    labels = [1,1,1,1,2,3,3,4,4,4]
    labels = create_labels(labels)
    cap = cv2.VideoCapture('data/000453.mp4')
    for i in range(100):
        _, frame = cap.read()
        cv2.putText(frame, str(labels[i]) + '-' + str(i+1), (20,60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        cv2.imshow('Frame', frame)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()

