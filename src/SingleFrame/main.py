#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataloader import get_loader
from model import SingleFrame
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
    batch_size = 32
    num_workers = 2
    # network parameters
    architecture = 'VGGNet11'
    pretrained = True
    finetuned = True
    # training parameters
    learning_rate = 1e-4
    max_epochs = 200
    criterion = nn.CrossEntropyLoss()

    # get dataloader
    dataloader, dataset_size = get_loader(data_path, batch_size, num_workers)
    print('Dataset Size:', dataset_size)
    # create network object
    net = SingleFrame(architecture, pretrained, finetuned)
    print(net)
    # create optimizer
    if not finetuned:
        optimizer = torch.optim.Adam(net.fc.parameters(), learning_rate)
    else:
        optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    # train the network
    best_acc, losses, accuracies = train_network(net, dataloader, dataset_size,
            batch_size, criterion, optimizer, max_epochs, gpu)
    # plot statistics
    print('Best Training Accuracy:', best_acc*100)
#    plot_data(losses, accuracies)

if __name__ == '__main__':
    main()

