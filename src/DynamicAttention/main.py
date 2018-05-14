#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataloader import get_loader
from model import DynamicAttention
from train import train_network

def plot_data(losses, accuracies):
    """Plot training statistics.

    Args:
        losses (list):      loss per training epoch
        accuracies (list):  accuracy per training epoch
    """
    import matplotlib.pyplot as plt

    # convert accuracies to percentages
    accuracies = [acc * 100 for acc in accuracies]
    plt.figure()
    plt.subplot(121), plt.plot(losses)
    plt.ylim(0, 2)
    plt.title('Training Losses')
    plt.subplot(122), plt.plot(accuracies)
    plt.ylim(0, 100)
    plt.title('Training Accuracy')
    plt.show()

def main():
    """Main Function."""
    # dataloader parameters
    gpu = torch.cuda.is_available()
    data_path = 'data/labels_done.txt'
    batch_size = 1
    num_workers = 0
    window_size = 10
    # network parameters
    architecture = 'AlexNet'
    hidden_size = 8
    rnn_layers = 1
    pretrained = False
    finetuned = False
    # training parameters
    learning_rate = 1e-4
    max_epochs = 2
    criterion = nn.CrossEntropyLoss()

    # get dataloader
    dataloader, dataset_size = get_loader(data_path, window_size, batch_size, 
            num_workers)
    print('Dataset Size:', dataset_size)

    # create network object
    net = DynamicAttention(architecture, batch_size, hidden_size, rnn_layers, 
            pretrained, finetuned)
    # create optimizer
    if not finetuned:
        p = list(net.embedding.parameters()) + list(net.attn.parameters()) \
                + list(net.attn_combine.parameters()) \
                + list(net.lstm.parameters()) + list(net.fc.parameters())
        optimizer = torch.optim.Adam(p, learning_rate)
    else:
        optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    # train the network
    best_acc, losses, accuracies = train_network(net, dataloader, dataset_size,
            criterion, optimizer, max_epochs, gpu)
    # plot statistics
#    print('Best Training Accuracy:', best_acc*100)
#    plot_data(losses, accuracies)

if __name__ == '__main__':
    main()

