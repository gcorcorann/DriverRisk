#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataloader import get_loader
from model import SingleStream
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
    batch_size = 4
    num_workers = 2
    window_size = 20
    # network parameters
    architecture = 'VGGNet11'
    rnn_hidden = 128
    rnn_layers = 1
    pretrained = True
    finetuned = True
    # training parameters
    learning_rate = 1e-4
    max_epochs = 200
    criterion = nn.CrossEntropyLoss()

    # get dataloader
    dataloader, dataset_size = get_loader(data_path, window_size, batch_size, 
            num_workers)
    print('Dataset Size:', dataset_size)
    # create network object
    net = SingleStream(architecture, rnn_hidden, rnn_layers, pretrained, 
            finetuned)
    # create optimizer
    if not finetuned:
        optimizer = torch.optim.Adam(
                list(net.lstm.parameters()) + list(net.fc.parameters()),
                learning_rate
                )
    else:
        optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    # train the network
    best_acc, losses, accuracies = train_network(net, dataloader, dataset_size,
            batch_size, window_size, criterion, optimizer, max_epochs, gpu)
    # plot statistics
    print('Best Training Accuracy:', best_acc*100)
    plot_data(losses, accuracies)

if __name__ == '__main__':
    main()

