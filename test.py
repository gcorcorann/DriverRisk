import cv2

labels = []
cap = cv2.VideoCapture('data/000453.mp4')
for i in range(1, 101):
    _, frame = cap.read()
    cv2.imshow('Frame', frame)
    cv2.waitKey(int(1/20*1000))
    if i % 10 == 0:
        lb = input('Danger Level: ')
        labels.append(lb)

print(labels)
