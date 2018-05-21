import torch
import torch.nn as nn
from dataloader2 import get_loader
from model2 import SingleStream, DynamicAttention
from train2 import train_network
import matplotlib.pyplot as plt

# set seed for reproducibility
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def plot_data(losses, accuracies):
    accuracies = [acc * 100 for acc in accuracies]

    plt.figure()
    plt.subplot(121)
    plt.plot(losses)
    plt.ylim(0, 2)
    plt.title('Training Losses')

    plt.subplot(122)
    plt.plot(accuracies)
    plt.ylim(0, 100)
    plt.title('Training Accuracy')
    plt.show()

def main():
    """Main Function."""
    # dataloader parameters
    data_path = 'data/labels2.txt'
    batch_size = 2
    num_workers = 1
    # network parameters
    hidden_size = 512
    rnn_layers = 2
    pretrained = True
    # training parameters
    learning_rate = 1e-4
    max_epochs = 100
    criterion = nn.CrossEntropyLoss()

    # get dataloader
    dataloader, dataset_size = get_loader(data_path, batch_size, num_workers,
            shuffle=True)
    print('Dataset Size:', dataset_size)

    # create network object
    net = DynamicAttention(hidden_size, rnn_layers, pretrained)
    # create optimizer
    p = list(net.embedding.parameters()) + list(net.attn.parameters()) \
            + list(net.attn_combine.parameters()) \
            + list(net.lstm.parameters()) + list(net.fc.parameters())
    optimizer = torch.optim.Adam(p, learning_rate)

    # train the network
    best_acc, losses, accuracies = train_network(net, dataloader,
            dataset_size, criterion, optimizer, max_epochs)
    # plot statistics
    print('Best Training Accuracy:', round(best_acc*100,2))
    plot_data(losses, accuracies)

if __name__ == '__main__':
    main()

