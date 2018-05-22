import torch
import torch.nn as nn
from dataloader2 import get_loaders
from model2 import SingleStream, DynamicAttention
from train2 import train_network
import matplotlib.pyplot as plt

# set seed for reproducibility
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def plot_data(losses, accuracies):
    # convert accuracies from decimals to percentages
    accuracies['Train'] = [acc * 100 for acc in accuracies['Train']]
    accuracies['Valid'] = [acc * 100 for acc in accuracies['Valid']]

    plt.figure()
    plt.subplot(121)
    plt.plot(losses['Train'], label='Training')
    plt.plot(losses['Valid'], label='Validation')
    plt.ylim(0, 2), plt.legend(loc='upper right')
    plt.title('Losses')

    plt.subplot(122)
    plt.plot(accuracies['Train'], label='Training')
    plt.plot(accuracies['Valid'], label='Validation')
    plt.ylim(0, 100), plt.legend(loc='upper left')
    plt.title('Accuracy')

    plt.show()

def main():
    """Main Function."""
    # dataloader parameters
    train_path = 'data/train_data.txt'
    valid_path = 'data/valid_data.txt'
    batch_size = 2
    num_workers = 1
    # network parameters
    hidden_size = 512
    rnn_layers = 2
    pretrained = True
    # training parameters
    learning_rate = 1e-4
    max_epochs = 20
    criterion = nn.CrossEntropyLoss()

    # get dataloaders
    dataloaders, dataset_sizes = get_loaders(train_path, valid_path, 
            batch_size, num_workers, shuffle=True)
    print('Dataset Sizes:', dataset_sizes)

    # create network object
    net = DynamicAttention(hidden_size, rnn_layers, pretrained)
    # create optimizer
    p = list(net.embedding.parameters()) + list(net.attn.parameters()) \
            + list(net.attn_combine.parameters()) \
            + list(net.lstm.parameters()) + list(net.fc.parameters())
    optimizer = torch.optim.Adam(p, learning_rate)

    # train the network
    best_acc, losses, accuracies = train_network(net, dataloaders,
            dataset_sizes, criterion, optimizer, max_epochs)
    # plot statistics
    print('Best Validation Accuracy:', round(best_acc*100,2))
    plot_data(losses, accuracies)

if __name__ == '__main__':
    main()

