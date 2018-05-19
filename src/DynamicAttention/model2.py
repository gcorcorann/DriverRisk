import torch.nn as nn
import torchvision.models as models

class SingleStream(nn.Module):
    """Single stream model.
    Args:
        rnn_hidden (int):   number of hidden units in each rnn layer
        rnn_layers (int):   number of layers in rnn model
        pretrained (bool):  if model is pretrained with ImageNet
    """

    def __init__(self, rnn_hidden, rnn_layers, pretrained):
        super().__init__()
        self.cnn = models.alexnet(pretrained)
        num_fts = self.cnn.classifier[4].in_features
        self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-3]
                )

        for param in self.cnn.parameters():
            param.requires_grad = False

        # add lstm layer
        self.lstm = nn.LSTM(num_fts, rnn_hidden, rnn_layers)
        # add fc layer
        self.fc = nn.Linear(rnn_hidden, 4)
        
    def forward(self, inp):
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

def main():
    """Test Function."""
    import torch

    rnn_hidden = 8
    rnn_layers = 2
    net = SingleStream(rnn_hidden, rnn_layers, pretrained=False)
    print(net)

    inputs = torch.randn(100, 2, 3, 224, 224)
    print('inputs:', inputs.shape)
    output = net.forward(inputs)
    print('output:', output.shape)

    

if __name__ == '__main__':
    main()

