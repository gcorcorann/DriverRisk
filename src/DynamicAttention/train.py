#!/usr/bin/env python3
import time
import torch
import copy
from torch.autograd import Variable

def train_network(net, dataloaders, dataset_sizes, criterion, optimizer, 
        max_epochs, gpu):
    """Train network.

    Args:
        net (torchvision.models):                   network to train
        dataloader (torch.utils.data.DataLoader):   dataloaders
        dataset_size (int):                         size of datasets
        criterion (torch.nn.modules.loss):          loss function
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
    losses = {'Train': [], 'Valid': []}
    accuracies = {'Train': [], 'Valid': []}
    best_acc = 0 
    best_model_wts = copy.deepcopy(net.state_dict())
    # when to stop training
    patience = 0
    for epoch in range(max_epochs):
        print()
        print('Epoch', epoch)
        print('-' * 8)
        # validation phase every x epochs
        if (epoch+1) % 5 == 0:
            phases = ['Train', 'Valid']
        else:
            phases = ['Train']

        # for each phase
        for phase in phases:
            if phase == 'Train':
                net.train()   # set network to training mode
            else:
                net.eval()    # set network to evaluate mode
        
            # used for losses + accuracies
            running_loss = 0
            running_correct = 0
            # iterate over data
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs and labels, reshape [seq_len, batch_size, *]
                inp_frames= data['X_frames'].transpose(0,1).to(device)
                inp_objs = data['X_objs'].transpose(0,1).to(device)
                labels = data['y'].transpose(0,1).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in training
                with torch.set_grad_enabled(phase == 'Train'):
                    if phase == 'Train':
                        # window size, batch size
                        ws, bs = inp_frames.shape[:2]
                    else:
                        ws, bs = 10, 10
                        # reshape data
                        inp_frames = inp_frames.view(ws, bs, 3, 224, 224)
                        inp_objs = inp_objs.view(ws, bs, 20, 3, 224, 224)
                        labels = labels.view(ws, bs)

                    # initialize hidden states
                    state = net.init_hidden(bs, device)
                    # for each timestep
                    for i in range(ws):
                        frame = inp_frames[i]
                        objs = inp_objs[i]
                        output, state, _ = net.forward(frame, objs, state,
                                device)

                    # loss + prediction
                    loss = criterion(output, labels[-1])
                    _, pred = torch.max(output, 1)
                    correct = (pred == labels[-1]).sum().item()
                    if phase == 'Train':
                        # backwards + optimize
                        loss.backward()
                        optimizer.step()
                        # statistics
                        running_loss += loss.item() * bs
                        running_correct += correct
                    else:
                        running_loss += loss.item()
                        running_correct += correct / bs
                    
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, 
                epoch_acc))
            # store stats
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            patience += 1
#TODO uncomment patience
#        if epoch_acc > best_acc:
#            best_acc = epoch_acc
#            best_model_wts = copy.deepcopy(net.state_dict())
#            patience = 0
#
#        if patience == 200:
#            break

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    #TODO remove line under
#    best_model_wts = copy.deepcopy(net.state_dict())
    net.load_state_dict(best_model_wts)
    # save to disk
    torch.save(net.state_dict(), 'data/model_params.pkl')

    return best_acc, losses, accuracies

