#!/usr/bin/env python3
from dataloader import get_loader
from model import DynamicAttention

def main():
    data_path = 'data/labels_done.txt'
    window_size = 100
    batch_size = 1
    num_workers = 0

    dataloader, dataset_size = get_loader(data_path, window_size, batch_size,
            num_workers)
    print('Dataset size:', dataset_size)

    batch = next(iter(dataloader))
    X_frames, X_objs, labels = batch['X_frames'], batch['X_objs'], batch['y']
    print('X_frames:', X_frames.shape)
    print('X_objs:', X_objs.shape)
    print('labels:', labels.shape)

    # create model
#    model = 'AlexNet'
#    batch_size = 1
#    hidden_size = 512
#    rnn_layers = 2
#    net = DynamicAttention(model, batch_size, hidden_size, rnn_layers,
#            pretrained=False, finetuned=False)
#    print(net)

    for i in range(10, 110, 10):
        
        print(i)
    


if __name__ == '__main__':
    main()

