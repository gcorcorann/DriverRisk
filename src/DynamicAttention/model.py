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
#        print('hidden:', hidden.shape)
#        print('objs:', objs.shape)
        batch_size = objs.shape[0]
        num_objs = objs.shape[1]
        attn_energies = self.score(hidden, objs)
#        print('attn_energies:', attn_energies.shape)
        # normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, objs):
        hidden = hidden.unsqueeze(2)
#        print('hidden:', hidden.shape)
        energy = self.attn(objs)
#        print('energy:', energy.shape)
        energy = torch.bmm(energy, hidden).squeeze(2)
#        print('energy:', energy.shape)
        return energy

class DynamicAttention(nn.Module):
    """Single stream model.
    Args:
        model (string):     CNN architecture
        hidden_size (int):  number of hidden units
        rnn_layers (int):   number of layers in rnn model
        pretrained (bool):  if model is pretrained with ImageNet (default true)
        finetuned (bool):   if model is finetuned (default true)
    """

    def __init__(self, model, hidden_size, rnn_layers, 
            pretrained=True, finetuned=True):
        super().__init__()
        # define parameters
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
        batch_size, num_objs = inp_objs.shape[0], inp_objs.shape[1]
#        print('inp_frame:', inp_frame.shape)
#        print('inp_objs:', inp_objs.shape)
        # break state into hidden state and cell state
        h, c = state
        # pass through CNN + embedding
        emb_frame = self.embedding(self.cnn.forward(inp_frame))
#        print('emb_frame:', emb_frame.shape)
        inp_objs = inp_objs.contiguous().view(-1, 3, 224, 224)
        emb_objs = self.embedding(self.cnn.forward(inp_objs))
        emb_objs = emb_objs.view(batch_size, num_objs, -1)
#        print('emb_objs:', emb_objs.shape)

        # for attention for each object (use last hidden layer)
        attn_weights = self.attn(h[-1], emb_objs, device)
#        print('attn_weights:', attn_weights.shape)

        # context
        context = torch.bmm(attn_weights, emb_objs).squeeze(1)
#        print('context:', context.shape)

        # combine
#        print('emb_frame:', emb_frame.shape)
        output = torch.cat((emb_frame, context), 1)
#        print('output:', output.shape)
        output = self.attn_combine(output).unsqueeze(0)
#        print('output:', output.shape)

        # pass through RNN
        output, state = self.lstm(output, state)
#        print('output:', output.shape)

        # pass through fc layer
        output = self.fc(output).squeeze(0)
#        print('output:', output.shape)
        return output, state, attn_weights.squeeze(1)


    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.rnn_layers, batch_size, self.hidden_size)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        return (h_0, c_0)

def main():
    """Test Function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = 'AlexNet'
    batch_size = 2
    hidden_size = 512
    rnn_layers = 2
    net = DynamicAttention(model, batch_size, hidden_size, rnn_layers, 
            pretrained=False, finetuned=False).to(device)
    print(net)

    window_size = 10
    max_objects = 20
    X_frames = torch.randn(window_size, batch_size, 3, 224, 224)
    X_objs = torch.randn(window_size, batch_size, max_objects, 3, 224, 224)
    X_frames = X_frames.to(device)
    X_objs = X_objs.to(device)
    print('X_frames:', X_frames.shape)
    print('X_objs:', X_objs.shape)

    # initialize hidden
    state = net.init_hidden(device)
    # for each time step
    for i in range(window_size):
        frame = X_frames[i]
        objs = X_objs[i]
        output, state, attn = net.forward(frame, objs, state, device)
        print('output:', output.shape)
        print('attn:', attn.shape)

if __name__ == '__main__':
    main()

