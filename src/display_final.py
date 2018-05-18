#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import glob
import random
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

def draw_objects(frame, objs, attn, color_map):
    attn = 1 - attn
    attn *= 255
    attn = attn.astype(np.uint8)
    attn = cv2.applyColorMap(attn, color_map)
    blend = 0.8
    for i, bb in enumerate(objs):
        left, top, right, bot = bb
        a = attn[i][0].astype(float)
        # alpha blending
        frame[top:bot, left:right] = a * blend + (1-blend) * frame[top:bot,
                left:right]

def draw_legend(w, h, color_map):
    legend = np.ones((h, 300, 3), dtype=np.uint8) * 255
    # color scale
    r = np.arange(1, h*7//9+1)
    r = r / np.max(r) * 255
    r = r.astype(np.uint8)
    r = cv2.applyColorMap(r, color_map)
    legend[h//9:h*8//9, 300//5:300*4//5] = r
    # words + bounding box
    cv2.rectangle(legend, (300//5,h//9), (300*4//5,h*8//9), (0,0,0), 2)
    cv2.putText(legend, 'High Attention', (40,h//11), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(legend, 'Low Attention', (40,h*16//17), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    return legend

def draw_plots(h, w, probs_list):
    grid = np.ones((260, w, 3), dtype=np.uint8) * 255

    text_size = 0.75
    thickness = 2
    color = (0,0,0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # y-axis
    cv2.arrowedLine(grid, (130,240), (130,10), color, thickness, 
            tipLength=0.05)
    # y-axis labels
    cv2.putText(grid, '100%', (45, 35), font, text_size, color, thickness)
    cv2.putText(grid, '50%', (60, 130), font, text_size, color, thickness)
    cv2.putText(grid, '0%', (70, 225), font, text_size, color, thickness)

    # y-axis ticks
    cv2.line(grid, (120,125), (140,125), color, thickness)
    cv2.line(grid, (120,30), (140,30), color, thickness)
    # x-axis
    cv2.arrowedLine(grid, (110, 220), (w-110, 220), color, thickness,
            tipLength=0.01)
    # x-axis label
    cv2.putText(grid, 'time', (w//2, 245), font, text_size, color, thickness)

    if len(probs_list)*20 >= w-110*2:
        probs_list.pop(0)

    draw_labels(grid, probs_list)

    return grid

def draw_labels(grid, probs_list):
    color_list = [(150, 215, 215), (0, 255, 255), (0, 100, 255), (0, 0, 255)]
    probs = np.array(probs_list)
    probs = np.int32(190*(1-probs) + 30)
    for i in range(4):
        pts = probs[:, i]
        t = np.arange(len(pts)) * 20 + 130
        pts = np.vstack((t, pts)).T
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(grid, [pts], False, color_list[i], 2)

def main():
    """Main Function."""
    fps = 10
    color_map = cv2.COLORMAP_HOT
    vid_path = 'data/videos/positive/000155.mp4'
    print('Video Path:', vid_path)

    objs_path = 'data/objects/' + vid_path[12:-3] + 'txt'
    with open(objs_path, 'r') as f:
        objs_data = f.read().split('\n')[:-1]

    with open('outputs_positive_000155.txt', 'r') as f:
        out_data = f.read().split('\n')[:-1]

    attn_scale = [1, 5, 50, 100]
    #attn_scale = [0.25, 0.50, 0.75, 1.0]
    probs_list = []
    # open video
    cap = cv2.VideoCapture(vid_path)
    for i in range(100):
        _, frame = cap.read()
        h, w = frame.shape[:2]
        # inputs from algorithm
        frame_data = np.float32(out_data[i].split())
        probs = frame_data[:4]
        #probs = F.softmax(torch.randn(1, 4).squeeze(), dim=0).numpy()
        print(probs)
        # store in list
        probs_list.append(probs)
        # label
        y_pred = np.argmax(probs)
        # attention
        attn = frame_data[4:]
#        attn = torch.randn(1, 20)
#        attn = attn.sigmoid()
        #TODO attention levels are mostly all zeros
        attn *= attn_scale[y_pred]
        print(attn)

        # draw labels graph
        graph = draw_graph(w,h,y_pred+1)
        # draw YOLO objects
        objs = np.array(objs_data[i].split(), dtype=int).reshape(-1, 4)
        draw_objects(frame, objs, attn, color_map)
        # draw scale legend
        legend = draw_legend(w, h, color_map)
        frame = np.hstack((legend, frame, graph))

        # draw probability plots
        plot = draw_plots(*frame.shape[:2], probs_list)
        frame = np.vstack((frame, plot))
        cv2.imshow('Frame', frame)
        #cv2.waitKey(int(1/fps*1000))
        cv2.waitKey(0)

if __name__ == '__main__':
    main()

