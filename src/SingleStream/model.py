#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models

class SingleStream(nn.Module):
    def __init__(self, hidden_size, rnn_layers, pretrained):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        # cnn
        self.cnn = models.alexnet(pretrained)
        num_fts = self.cnn.classifier[4].in_features
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-3]
                )

        for param in self.cnn.parameters():
            param.requires_grad = False

        # add lstm layer
        self.lstm = nn.LSTM(num_fts, hidden_size, rnn_layers)
        # add fc layer
        self.fc = nn.Linear(hidden_size, 4)
        
    def forward_batch(self, inp):
        sequence_len, batch_size = inp.shape[:2]
        # reshape for CNN [batch_size*sequence_len, num_channels, width, height]
        inp = inp.contiguous().view(-1, *inp.shape[2:])
        # pass through CNN
        outs = self.cnn.forward(inp)
        # reshape back to [sequence_len, batch_size, num_feats]
        outs = outs.view(sequence_len, batch_size, -1)
        # pass through LSTM
        outs, _ = self.lstm(outs)
        # pass through fully-connected
        outs = self.fc(outs)
        return outs

    def forward(self, inp, states):
        out = self.cnn(inp)
        out, states = self.lstm(out.unsqueeze(0), states)
        out = self.fc(out).squeeze(0)
        return out, states

    def init_states(self, batch_size, device):
        h_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        return (h_0, c_0)

def main():
    """Test Function."""
    hidden_size = 8
    rnn_layers = 2
    net = SingleStream(hidden_size, rnn_layers, pretrained=False)
    print(net)

if __name__ == '__main__':
    main()

