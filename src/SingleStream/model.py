#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as models

class SingleStream(nn.Module):
    def __init__(self, model, hidden_size, rnn_layers, pretrained, finetuned):
        super().__init__()
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
        elif model is 'ResNet50':
            self.cnn = models.resnet50(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        elif model is 'ResNet101':
            self.cnn = models.resnet101(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        elif model is 'ResNet152':
            self.cnn = models.resnet152(pretrained)
            num_fts = self.cnn.fc.in_features
            self.cnn = nn.Sequential(
                    *list(self.cnn.children())[:-1]
                    )
        elif model is 'Inception':
            self.cnn = models.inception_v2(pretrained)
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
        self.lstm = nn.LSTM(num_fts, hidden_size, rnn_layers)
        # add fc layer
        self.fc = nn.Linear(hidden_size, 4)
        
    def forward_batch(self, inp):
        #TODO introduce dropout layers to help with generalization
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
    model = 'AlexNet'
    hidden_size = 8
    rnn_layers = 2
    net = SingleStream(model, hidden_size, rnn_layers, pretrained=False,
            finetuned=False)
    print(net)

if __name__ == '__main__':
    main()

