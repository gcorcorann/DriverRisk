#!/usr/bin/env python3
import time
import torch
import torch.nn.functional as F
from dataloader import get_loader
from model import DynamicAttention

def main():
    """MAIN FUNCTION."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = 'data/labels_done.txt'
    batch_size = 1
    num_workers = 0
    sequence_len = 100
    ws = 10  # window_size

    dataloader, dataset_size = get_loader(data_path, sequence_len, batch_size,
            num_workers)
    print('Dataset size:', dataset_size)

    it = iter(dataloader)
    for _ in range(10):
        batch = next(it)

    X_frames = batch['X_frames'].transpose(0, 1).to(device)
    X_objs = batch['X_objs'].transpose(0, 1).to(device)
    print('X_frames:', X_frames.shape)
    print('X_objs:', X_objs.shape)

    # create model
    model = 'VGGNet11'
    batch_size = 1
    hidden_size = 512
    rnn_layers = 2
    net = DynamicAttention(model, batch_size, hidden_size, rnn_layers,
            pretrained=True, finetuned=False).to(device)
    net.load_state_dict(torch.load('data/model_params.pkl'))

    with open('outputs_positive_000512.txt', 'w') as f:
        for i in range(1, sequence_len+1):
            inp_frames = X_frames[i-ws if i-ws > 0 else 0: i]
            inp_objs = X_objs[i-ws if i-ws > 0 else 0: i]
            print('inp_frames:', inp_frames.shape)
            print('inp_objs:', inp_objs.shape)
    
            state = net.init_hidden(device)
    
            start_time = time.time()
            # for each timestep
            for t in range(inp_frames.shape[0]):
                frame = inp_frames[t]
                objs = inp_objs[t]
                output, state, attn = net.forward(frame, objs, state, device)
    
            output = F.softmax(output.squeeze(), dim=0)
            attn = attn.squeeze()
            print('output:', output.shape)
            print('attn:', attn.shape)
            output = output.tolist()
            attn = attn.tolist()
            for item in output:
                f.write(str(item) + ' ')
            for item in attn:
                f.write(str(item) + ' ')
            f.write('\n')
            fps = 1 / ((time.time() - start_time) + 1/30)
            print('FPS:', fps)

if __name__ == '__main__':
    main()

