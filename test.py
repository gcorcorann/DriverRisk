#!/usr/bin/env python3
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

class SingleStream(nn.Module):
    """Single stream model.
    Args:
        model (string):     CNN architecture
        rnn_hidden (int):   number of hidden units in each rnn layer
        rnn_layers (int):   number of layers in rnn model
        pretrained (bool):  if model is pretrained with ImageNet (default true)
        finetuned (bool):   if model is finetuned (default true)
    """

    def __init__(self, model, rnn_hidden, rnn_layers, pretrained=True, 
            finetuned=True):
        super().__init__()
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

        # add lstm layer
        self.lstm = nn.LSTM(num_fts, rnn_hidden, rnn_layers, batch_first=True)
        # add fc layer
        self.fc = nn.Linear(rnn_hidden, 4)
        
    def forward(self, inp):
        # grab batch size for reshaping tensor
        batch_size = inp.shape[0]
        # reshape into [batch, numChannels, Height, Width]
        inp = inp.view(-1, *inp.shape[2:])
        # pass through CNN
        outs = self.cnn.forward(inp)
        # reformat back to [batchSize, numSeq, numFeats]
        outs = outs.view(batch_size, -1, outs.shape[1])
        # pass through LSTM
        outs, _ = self.lstm(outs)
        # pass through fully-connected
        outs = self.fc(outs[0, -1])
        return outs

def main():
    """Test Function."""
    device = torch.device("cuda")
    # DATA
    data_path = 'data/processed/positive/000550.npy'
    X = np.load(data_path)
    X = X[:20, :244, :224]
    X = X.transpose(0,3,1,2)
    X = torch.from_numpy(X).unsqueeze(0).type(torch.float)
    print(X.shape)
    X_new = [X for i in range(20)]
    X = torch.cat(X_new, dim=1)
    print(X.shape)
    X = X.to(device)

    # MODEL
    architecture = 'VGGNet11'
    rnn_hidden = 128
    rnn_layers = 2
    net = SingleStream(architecture, rnn_hidden, rnn_layers, pretrained=False, 
            finetuned=False)
    net = net.to(device)

    start_time = time.time()
    outputs = net.forward(X)
    fps = round(1 / (time.time() - start_time), 1)
    print('FPS: ', fps)
    print('outputs:', outputs.shape)


if __name__ == '__main__':
    main()
