import time
import torch
import copy

def train_network(net, dataloaders, dataset_sizes, criterion, optimizer, 
        max_epochs):
    """Train network.

    Args:
        net (torchvision.models):                   network to train
        dataloaders (torch.utils.data.DataLoader):  dataloaders
        dataset_sizes (int):                        size of datasets
        criterion (torch.nn.modules.loss):          loss function
        optimizer (torch.optim):    optimization algorithm
        max_epochs (int):           maximum number of epochs used for training

    Returns:
        best_acc (int):     best validation accuracy
        losses (list):      losses
        accuracies (list):  accuracies
    """
    # start timer
    start = time.time()
    # cpu/gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # store network to cpu/gpu
    net = net.to(device)
    # training losses + accuracies
    losses = {'Train': [], 'Valid': []}
    accuracies = {'Train': [], 'Valid': []}
    best_acc = 0 
    best_net_wts = copy.deepcopy(net.state_dict())
    # when to stop training
    patience = 0
    # for each epoch
    for epoch in range(max_epochs):
        print()
        print('Epoch', epoch)
        print('-' * 8)
        # each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                net.train() # set network to training mode
            else:
                net.eval()  # set network to evaluation mode

            # used for losses + accuracies
            running_loss = 0
            running_correct = 0
            # iterate over data
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs and labels, reshape [seq_len, batch_size, *]
                X_frames, X_objs, y = data
                X_frames = X_frames.transpose(0, 1).to(device)
                X_objs = X_objs.transpose(0, 1).to(device)
                y = y.transpose(0, 1).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # input size
                sequence_len, batch_size = X_frames.shape[:2]
                # initialize hidden states
                states = net.init_states(batch_size, device)
                # for each timestep
                loss = 0
                correct = 0
                # track history if in training phase
                with torch.set_grad_enabled(phase == 'Train'):
                    for t in range(sequence_len):
                        frame = X_frames[t]
                        objs = X_objs[t]
                        # forward pass
                        output, states, _ = net.forward(frame, objs, states)
                        # loss + prediction
                        loss += criterion(output, y[t])
                        _, y_pred = torch.max(output, 1)
                        correct += (y_pred == y[t]).sum().item()

                    # backwards + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_size / sequence_len
                running_correct += correct / sequence_len
                        
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, 
                epoch_acc))
            # store stats
            losses[phase].append(epoch_loss)
            accuracies[phase].append(epoch_acc)
            # if in validation phase
            if phase == 'Valid':
                patience += 1  # increase patience
                # save best accuracy
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_net_wts = copy.deepcopy(net.state_dict())
                    patience = 0
    
                if patience == 10:
                    break

    # print elapsed time
    time_elapsed = time.time() - start
    print()
    print('Training Complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # load best network
    net.load_state_dict(best_net_wts)
    return net, best_acc, losses, accuracies
