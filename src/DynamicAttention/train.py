#!/usr/bin/env python3
import time
import torch
import copy
from torch.autograd import Variable

def train_network(net, dataloader, dataset_size, criterion, optimizer, 
        max_epochs, gpu):
    """Train network.

    Args:
        net (torchvision.models):   network to train
        dataloader (torch.utils.data.DataLoader):   training dataloader
        dataset_size (int):         size of dataset
        criterion (torch.nn.modules.loss):      loss function
        optimizer (torch.optim):    optimization algorithm
        max_epochs (int):           maximum number of epochs used for training
        gpu (bool):                 presence of gpu

    Returns:
        best_acc (int):     best training accuracy
        losses (list):      training losses
        accuracies (list):  training accuracies
    """
    # start timer
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # store network to cpu/gpu
    net = net.to(device)
    # training losses + accuracies
    losses, accuracies = [], []
    best_acc = 0 
    best_model_wts = copy.deepcopy(net.state_dict())
    # when to stop training
    patience = 0
    for epoch in range(max_epochs):
        print()
        print('Epoch', epoch)
        print('-' * 8)
        # set to training model
        net.train(True)

        # used for losses + accuracies
        running_loss = 0
        running_correct = 0
        # iterate over data
        for i, data in enumerate(dataloader):
            # get the inputs and labels
            inp_frames= data['X_frames'].to(device)
            inp_objs = data['X_objs'].to(device)
            labels = data['y'].to(device)
            # reshape [seqLen, batchSize, *]
            inp_frames = inp_frames.transpose(0, 1)
            inp_objs = inp_objs.transpose(0, 1)
            labels = labels.transpose(0, 1)

            # zero the parameter gradients
            optimizer.zero_grad()
            # initiaize hidden states
            state = net.init_hidden(device)
            loss = 0
            correct = 0
            # for each time-step
            for i in range(inp_frames.shape[0]):
                frame = inp_frames[i]
                objs = inp_objs[i]
                output, state, _ = net.forward(frame, objs, state, device)

            # loss + predicted
            loss = criterion(output, labels[-1])
            _, pred = torch.max(output, 1)
            correct = (pred == labels[-1]).sum().item()

            # backwards + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inp_frames.shape[1]
            running_correct += correct
                    
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_correct / dataset_size
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
        # store stats
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        patience += 1
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())
            patience = 0

        if patience == 20:
            break

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    net.load_state_dict(best_model_wts)
    # save to disk
    torch.save(net.state_dict(), 'data/model_params.pkl')

    return best_acc, losses, accuracies

