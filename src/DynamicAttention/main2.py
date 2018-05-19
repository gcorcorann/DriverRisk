import torch
import torch.nn as nn
from dataloader import get_loader
from model import DynamicAttention
from train import train_network


def main():
    """Main Function."""
    # dataloader parameters
    gpu = torch.cuda.is_available()
    data_path = 'data/labels.txt'
    batch_size = 2
    num_workers = 2
    # network parameters
    hidden_size = 512
    rnn_layers = 2
    pretrained = True
    # training parameters
    learning_rate = 1e-4
    max_epochs = 2
    criterion = nn.CrossEntropyLoss()

    # get dataloader
    dataloader, dataset_size = get_loader(data_path, batch_size, num_workers)
    print('Dataset Size:', dataset_sizes)

    # create network object
    net = SingleStream(hidden_size, rnn_layers, pretrained)
    # create optimizer
    p = list(net.lstm.parameters()) + list(net.fc.parameters())
    optimizer = torch.optim.Adam(p, learning_rate)

    # train the network
    best_acc, losses, accuracies = train_network(net, dataloaders, 
            dataset_size, criterion, optimizer, max_epochs, gpu)
    # plot statistics
    print('Best Training Accuracy:', best_acc*100)

if __name__ == '__main__':
    main()

