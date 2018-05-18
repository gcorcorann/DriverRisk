#!/usr/bin/env python3
import torch
import torch.nn as nn
from dataloader import get_loader
from model import DynamicAttention
from train import train_network

def plot_data(losses, accuracies):
    """Plot training statistics.

    Args:
        losses (dict):      training and validation losses per epoch
        accuracies (dict):  training and validation accuracy per epoch
    """
    import matplotlib.pyplot as plt

    # convert accuracies to percentages
    accuracies['Train'] = [acc * 100 for acc in accuracies['Train']]
    accuracies['Valid'] = [acc * 100 for acc in accuracies['Valid']]
    plt.figure()
    plt.subplot(221)
    plt.plot(losses['Train'])
    plt.ylim(0, 2)
    plt.title('Training Losses')
    
    plt.subplot(222)
    plt.plot(losses['Valid'])
    plt.ylim(0, 2)
    plt.title('Validation Losses')

    plt.subplot(223)
    plt.plot(accuracies['Train'])
    plt.ylim(0, 100)
    plt.title('Training Accuracy')

    plt.subplot(224)
    plt.plot(accuracies['Valid'])
    plt.ylim(0, 100)
    plt.title('Training Accuracy')
    plt.show()

def main():
    """Main Function."""
    # dataloader parameters
    gpu = torch.cuda.is_available()
    data_path = 'data/labels.txt'
    batch_size = 2
    num_workers = 2
    window_size = 10
    # network parameters
    architecture = 'AlexNet'
    hidden_size = 512
    rnn_layers = 2
    pretrained = True
    finetuned = False
    # training parameters
    learning_rate = 1e-4
    max_epochs = 100
    criterion = nn.CrossEntropyLoss()

    # get dataloader
    dataloaders, dataset_sizes = get_loader(data_path, window_size, batch_size, 
            num_workers)
    print('Dataset Size:', dataset_sizes)

    # create network object
    net = DynamicAttention(architecture, hidden_size, rnn_layers, 
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
    best_acc, losses, accuracies = train_network(net, dataloaders, 
            dataset_sizes,
            criterion, optimizer, max_epochs, gpu)
    # plot statistics
    print('Best Training Accuracy:', best_acc*100)
    plot_data(losses, accuracies)

if __name__ == '__main__':
    main()

