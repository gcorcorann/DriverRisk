#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, objs, device):
        num_objs = objs.shape[1]
        attn_energies = torch.zeros(num_objs).to(device)
        # calculate energies for each encoder output
        for i in range(num_objs):
            attn_energies[i] = self.score(hidden, objs[:, i])

        # normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies, dim=0).unsqueeze(0)

    def score(self, hidden, obj):
        if self.method == 'dot':
            energy = hidden.dot(obj)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(obj)
            energy = energy.squeeze()
            hidden = hidden.squeeze()
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy

class DynamicAttention(nn.Module):
    """Single stream model.
    Args:
        model (string):     CNN architecture
        batch_size (int):   number of instances in mini-batch
        hidden_size (int):  number of hidden units
        rnn_layers (int):   number of layers in rnn model
        pretrained (bool):  if model is pretrained with ImageNet (default true)
        finetuned (bool):   if model is finetuned (default true)
    """

    def __init__(self, model, batch_size, hidden_size, rnn_layers, 
            pretrained=True, finetuned=True):
        super().__init__()
        # define parameters
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers

        if model is 'AlexNet':
            self.cnn = models.alexnet(pretrained)
            num_fts = self.cnn.classifier[4].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-3]
                    )
        elif model is 'VGGNet11':
            self.cnn = models.vgg11_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'VGGNet16':
            self.cnn = models.vgg16_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'VGGNet19':
            self.cnn = models.vgg19_bn(pretrained)
            num_fts = self.cnn.classifier[3].in_features
            self.cnn.classifier = nn.Sequential(
                    *list(self.cnn.classifier.children())[:-4]
                    )
        elif model is 'ResNet18':
            self.cnn = models.resnet18(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        elif model is 'ResNet34':
            self.cnn = models.resnet34(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        else:
            print('Please input correct model architecture')
            return

        for param in self.cnn.parameters():
            param.requires_grad = finetuned

        # define layers
        self.embedding = nn.Linear(num_fts, hidden_size)
        # attention layer
        self.attn = Attn('general', hidden_size)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        # add lstm layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, rnn_layers)
        # add fc layer
        self.fc = nn.Linear(hidden_size, 4)
        
    def forward(self, inp_frame, inp_objs, state, device):
        # break state into hidden state and cell state
        h, c = state
        # pass through CNN + embedding
        emb_frame = self.embedding(self.cnn.forward(inp_frame))
        window_size = inp_objs.shape[1]
        inp_objs = inp_objs.view(-1, 3, 224, 224)
        emb_objs = self.embedding(self.cnn.forward(inp_objs))
        emb_objs = emb_objs.view(self.batch_size, window_size, -1)

        # for attention for each object
        attn_weights = self.attn(h[-1], emb_objs, device)

        # context
        context = torch.mm(attn_weights, emb_objs[0])

        # combine
        output = torch.cat((emb_frame, context), 1)
        output = self.attn_combine(output)

        # pass through RNN
        output = output.unsqueeze(0)
        output, state = self.lstm(output, state)

        # pass through fc layer
        output = self.fc(output).squeeze(0)
        return output, state


    def init_hidden(self, device):
        h_0 = torch.zeros(self.rnn_layers, self.batch_size, self.hidden_size)
        c_0 = torch.zeros(self.rnn_layers, self.batch_size, self.hidden_size)
        h_0.to(device)
        c_0.to(device)
        return (h_0, c_0)

    def score(self, hidden, obj):
        energy = self.attn(obj)
        energy = energy.squeeze()
        hidden = hidden.squeeze()
        energy = hidden.dot(energy)
        return energy


def main():
    """Test Function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = 'AlexNet'
    batch_size = 1
    hidden_size = 512
    rnn_layers = 1
    net = DynamicAttention(model, batch_size, hidden_size, rnn_layers, 
            pretrained=False, finetuned=False)
    net.to(device)
    print(net)

    window_size = 10
    max_objects = 20
    X_frames = torch.randn(window_size, batch_size, 3, 224, 224)
    X_objs = torch.randn(window_size, batch_size, max_objects, 3, 224, 224)
    X_frames.to(device)
    X_objs.to(device)
    print('X_frames:', X_frames.shape)
    print('X_objs:', X_objs.shape)

    # initialize hidden
    state = net.init_hidden(device)
    # for each time step
    for i in range(window_size):
        frame = X_frames[i]
        objs = X_objs[i]
        output, state = net.forward(frame, objs, state, device)
        print('output:', output.shape)

if __name__ == '__main__':
    main()

