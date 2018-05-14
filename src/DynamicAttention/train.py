#!/usr/bin/env python3
import time
import torch
from torch.autograd import Variable

def train_network(net, dataloader, dataset_size, batch_size, window_size,
        criterion, optimizer, max_epochs, gpu):
    """Train network.

    Args:
        net (torchvision.models):   network to train
        dataloader (torch.utils.data.DataLoader):   training dataloader
        dataset_size (int):         size of dataset
        batch_size (int):           size of mini-batch
        window_size (int):          size of sliding window
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
    if gpu:
        # store network to gpu
        net = net.cuda()
    
    # training losses + accuracies
    losses, accuracies = [], []
    best_acc = 0 
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
            # get the inputs
            inp_frames, inp_objs = data['X_frames'], data['X_objs']
            labels = data['y']
            # reshape [seqLen, batchSize, *]
            inp_frames = inp_frames.transpose(0, 1)
            inp_objs = inp_objs.transpose(0, 1)
            print('inp_frames:', inp_frames.shape)
            print('inp_objs:', inp_objs.shape)
            print('labels:', labels.shape)
            if gpu:
                inp_frames = inp_frames.cuda()
                inp_objs = inp_objs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # initiaize hidden states
            # TODO move to cuda
            state = net.init_hidden()
            loss = 0
            correct = 0
            # for each time-step
            for i in range(window_size):
                frame = inp_frames[i]
                objs = inp_objs[i]
                output, state = net.forward(frame, objs, state)
                # loss + predicted
                loss += criterion(output, labels[:, i])
                _, pred = torch.max(output.detach(), 1)
                correct += torch.sum(pred == labels[:,i].detach())

            # backwars + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_correct += correct.item()

        epoch_loss = running_loss * batch_size / dataset_size 
        epoch_acc = running_correct / (dataset_size * window_size)
        print('Training Loss: {:.4f} Acc: {:.4f}'.format(
            epoch_loss, epoch_acc))
        # store stats
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        patience += 1
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            patience = 0

        if patience == 20:
            break

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_acc, losses, accuracies

