#!/usr/bin/env python3
import time
import torch
import torch.nn.functional as F
from dataloader2 import get_loader
from model2 import DynamicAttention

# set seed for reproducibility
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def main():
    """MAIN FUNCTION."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = 'data/labels2.txt'
    batch_size = 1
    num_workers = 0
    sequence_len = 100

    # get DataLoader object
    dataloader, dataset_size = get_loader(data_path, batch_size, num_workers,
            shuffle=False)
    print('Dataset size:', dataset_size)

    # get mini-batch
    it = iter(dataloader)
    for i, batch in enumerate(dataloader):
        # extract data
        X_frames, X_objs, y = batch
        X_frames = X_frames.transpose(0, 1).to(device)
        X_objs = X_objs.transpose(0, 1).to(device)
        y = y.transpose(0, 1).to(device)

        # create network
        hidden_size = 512
        rnn_layers = 2
        pretrained = True
        net = DynamicAttention(hidden_size, rnn_layers, pretrained).to(device)
        # load weights
        net.load_state_dict(torch.load('data/net_params.pkl'))
        # set to evaluation mode
        net = net.eval()

        # initialize hidden states
        states = net.init_states(batch_size, device)
        s = 'outputs/{}.txt'.format(i+1)
        with open(s, 'w') as f:
            # for each timestep
            for t in range(sequence_len):
                start_time = time.time()
                frame = X_frames[t]
                objs = X_objs[t]
                print('objs:', objs[0, :, 0, 100, 100])
                output, states, attn = net.forward(frame, objs, states)
                output = F.softmax(output.squeeze(), dim=0)
                attn = attn.squeeze()
                output = output.tolist()
                attn = attn.tolist()
                print('output:', output)
                print('attn:', attn)
                for item in output:
                    f.write(str(item) + ' ')
                for item in attn:
                    f.write(str(item) + ' ')
                f.write('\n')
                fps = 1 / ((time.time() - start_time) + 1/30)
                print('FPS:', fps)
                print()

if __name__ == '__main__':
    main()

