import torch
import torch.nn as nn
import torch.nn.functional as F
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
        inp = inp.view(-1, *inp.shape[2:])
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
        return out

    def init_states(self, batch_size, device):
        h_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        return (h_0, c_0)

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def forward(self, hidden, objs):
        batch_size, num_objs = objs.shape[:2]
        attn_energies = self.score(hidden, objs)
        # normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, objs):
        hidden = hidden.unsqueeze(2)
        energy = self.attn(objs)
        energy = torch.bmm(energy, hidden).squeeze(2)
        return energy

class DynamicAttention(nn.Module):
    def __init__(self, hidden_size, rnn_layers, pretrained=True):
        super().__init__()
        # define parameters
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

        # define layers
        self.embedding = nn.Linear(num_fts, hidden_size)
        # attention layer
        self.attn = Attn(hidden_size)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        # add lstm layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, rnn_layers)
        # add fc layer
        self.fc = nn.Linear(hidden_size, 4)

    def forward(self, inp_frame, inp_objs, states):
        batch_size, num_objs = inp_objs.shape[:2]
        # break state into hidden state and cell state
        h, c = states
        # pass through CNN + embedding
        emb_frame = self.embedding(self.cnn.forward(inp_frame))
        inp_objs = inp_objs.contiguous().view(-1, 3, 224, 224)
        emb_objs = self.embedding(self.cnn.forward(inp_objs))
        emb_objs = emb_objs.view(batch_size, num_objs, -1)
        # for attention for each object (use last hidden layer)
        attn_weights = self.attn(h[-1], emb_objs)
        # context
        context = torch.bmm(attn_weights, emb_objs).squeeze(1)
        # combine
        output = torch.cat((emb_frame, context), 1)
        output = self.attn_combine(output).unsqueeze(0)
        # pass through RNN
        output, states = self.lstm(output, states)
        # pass through fc layer
        output = self.fc(output).squeeze(0)
        return output, states, attn_weights.squeeze(1)

    def init_states(self, batch_size, device):
        h_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        return (h_0, c_0)

def main():
    """Test Function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = 512
    rnn_layers = 2
    net = DynamicAttention(hidden_size, rnn_layers, pretrained=False)
    #net = SingleStream(rnn_hidden, rnn_layers, pretrained=False)
    net = net.to(device)
    print(net)

    sequence_len, batch_size = 100, 2
    X_frames = torch.randn(sequence_len, batch_size, 3, 224, 224).to(device)
    X_objs = torch.randn(sequence_len, batch_size, 10, 3, 224, 224).to(device)
    print('X_frames:', X_frames.shape)
    print('X_objs:', X_objs.shape)

    # initialize hidden
    states = net.init_states(batch_size, device)
    # for each timestep
    for i in range(sequence_len):
        frame = X_frames[i]
        objs = X_objs[i]
        output, states, attn = net.forward(frame, objs, states)
        print('output:', output.shape)
        print('attn:', attn.shape)

if __name__ == '__main__':
    main()

