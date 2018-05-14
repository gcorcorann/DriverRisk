#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        # add lstm layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, rnn_layers)
        # add fc layer
        self.fc = nn.Linear(hidden_size, 4)
        
    def forward(self, inp_frame, inp_objs, state, device):
        # break state into hidden state and cell state
        h, c = state
#        print('inp_frame:', inp_frame.shape)
#        print('inp_objs:', inp_objs.shape)
#        print('h:', h.shape)
#        print('c:', c.shape)
        # pass through CNN + embedding
        emb_frame = self.embedding(self.cnn.forward(inp_frame))
#        print('emb_frame:', emb_frame.shape)
        window_size = inp_objs.shape[1]
        inp_objs = inp_objs.view(-1, 3, 224, 224)
        emb_objs = self.embedding(self.cnn.forward(inp_objs))
        emb_objs = emb_objs.view(self.batch_size, window_size, -1)
#        print('emb_objs:', emb_objs.shape)

        # for attention for each object
        num_objs = emb_objs.shape[1]
        attn_energies = torch.zeros(num_objs).to(device)
        # calculate energies for each object
        for i in range(num_objs):
            attn_energies[i] = self.score(h[-1], emb_objs[:, i])

#        print('attn_energies:', attn_energies.shape)
        # normalize energies
        attn_weights = F.softmax(attn_energies, dim=0).unsqueeze(0)
#        print('attn_weights:', attn_weights.shape)

        # context
        context = torch.mm(attn_weights, emb_objs[0])
#        print('context:', context.shape)

        # combine
        output = torch.cat((emb_frame, context), 1)
#        print('output:', output.shape)
        output = self.attn_combine(output)
#        print('output:', output.shape)

        # pass through RNN
        output = output.unsqueeze(0)
        output, state = self.lstm(output, state)
#        print('output:', output.shape)
#        print('h:', state[0].shape)
#        print('c:', state[1].shape)

        # pass through fc layer
        output = self.fc(output).squeeze(0)
#        print('output:', output.shape)
        return output, state


    def init_hidden(self, device):
        h_0 = torch.zeros(self.rnn_layers, self.batch_size, self.hidden_size)
        c_0 = torch.zeros(self.rnn_layers, self.batch_size, self.hidden_size)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        return (h_0, c_0)

    def score(self, hidden, obj):
        energy = self.attn(obj)
        energy = energy.squeeze()
        hidden = hidden.squeeze()
        energy = hidden.dot(energy)
        return energy


def main():
    """Test Function."""
    model = 'AlexNet'
    batch_size = 1
    hidden_size = 512
    rnn_layers = 1
    net = DynamicAttention(model, batch_size, hidden_size, rnn_layers, 
            pretrained=False, finetuned=False)
    print(net)

    window_size = 10
    max_objects = 20
    X_frames = torch.randn(window_size, batch_size, 3, 224, 224)
    X_objs = torch.randn(window_size, batch_size, max_objects, 3, 224, 224)
    print('X_frames:', X_frames.shape)
    print('X_objs:', X_objs.shape)

    # initialize hidden
    state = net.init_hidden()
    # for each time step
    for i in range(window_size):
        frame = X_frames[i]
        objs = X_objs[i]
        output, state = net.forward(frame, objs, state)
        print('output:', output.shape)



if __name__ == '__main__':
    main()

