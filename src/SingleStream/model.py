#!/usr/bin/env python3
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
        outs = self.fc(outs)
        return outs

def main():
    """Test Function."""
    model = 'VGGNet11'
    rnn_hidden = 8
    rnn_layers = 2
    net = SingleStream(model, rnn_hidden, rnn_layers, pretrained=False, 
            finetuned=False)
    print(net)

if __name__ == '__main__':
    main()

