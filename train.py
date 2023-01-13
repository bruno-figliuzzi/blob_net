"""
Training the neural network architecture
"""

import sys, time, os
from os.path import expanduser
home = expanduser("~")

import torch
import torchvision
import torchvision.transforms as transforms
from dataset import *
import torch.nn as nn
import torch.nn.functional as F
from network import Net

# -------------------------------------------------------------------
# STEP 1: Select parameters
# ------------------------------------------------------------------

def train(epoch):

    """
    Train the network

    Parameters
    ----------
    epoch: int
       current epoch
    """

    epoch_loss = []
    for i, sample in enumerate(trainloader):
                
        # Load batch
        input_img = sample['input'].to(device)
        segm_true = sample['target'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward/Backward pass
        outputs = net(input_img)
        loss = criterion(outputs, segm_true)
        loss.backward()
        epoch_loss.append(loss.item())

        # Weights update
        optimizer.step()

    # Write the training loss in a .dat file
    training_loss = np.mean(np.array(epoch_loss))
    print('Training loss: ' + str(training_loss))
    with open(os.path.join(THIS_FOLDER, 'training_loss.dat'), "a") as f:
        f.write('%d %.15f\n' % (epoch + 1, training_loss))

    # Save weights
    if(epoch % 10 == 0):
        torch.save(net.state_dict(), 
          os.path.join(weights_path,str(epoch//10) + '.pth'))


def test(epoch):

    """
    Test the network

    Parameters
    ----------
    epoch: int
       current epoch
    """
    epoch_loss = []
    for i, sample in enumerate(testloader):
                
        # Load batch
        input_img = sample['input'].to(device)
        segm_true = sample['target'].to(device)

        # Forward/Backward pass
        outputs = net(input_img)
        loss = criterion(outputs, segm_true)
        epoch_loss.append(loss.item())

    validation_loss = np.mean(np.array(epoch_loss))
    print('Validation loss: ' + str(validation_loss))
    with open(os.path.join(THIS_FOLDER, 'test_loss.dat'), "a") as f:
        f.write('%d %.15f\n' % (epoch + 1, loss.item()))


# -------------------------------------------------------------------
# STEP 2: Launch training
# -------------------------------------------------------------------

if __name__ == '__main__':

    # Paths
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(THIS_FOLDER,'dataset')
    train_dir = 'train'
    test_dir = 'val'
    target_dir = 'labels'

    # Select a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # Parameters
    d = 7
    w = 24
    dict_weights = None
    batch_size = 8
    start_epoch = 0
    num_epochs = 50
    learning_rate = 0.01
    divide = 2.  
    each = 50

    # Training set
    train_dataset = SegmentationDataset(
       root_dir= root_dir,
       input_dir= train_dir,
       target_dir= target_dir,
       transform=transforms.Compose([
	     RandomCrop(128),
	     Distort(),
	     SetTarget(),
	     Normalize(),
	     ToTensor()]))

    trainloader = torch.utils.data.DataLoader(
       train_dataset,
       batch_size = batch_size,
       shuffle = True,
       num_workers = 0)

    # Test set
    test_dataset = SegmentationDataset(
       root_dir= root_dir,
       input_dir= test_dir,
       target_dir= target_dir,
       transform=transforms.Compose([
	     RandomCrop(128),
	     SetTarget(),
	     Normalize(),
	     ToTensor()]))

    testloader = torch.utils.data.DataLoader(
       test_dataset,
       batch_size = batch_size,
       num_workers = 0)


    # Constructs the neural network
    net = Net(d, w)
    net.to(device)
    if(dict_weights != None):
        net.load_state_dict(torch.load(dict_weights))

    weights_path = os.path.join(THIS_FOLDER, 'weights')
    loss_path = os.path.join(THIS_FOLDER, 'output')
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

    # Optimizer
    import torch.optim as optim
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)

    for epoch in range(start_epoch, num_epochs):

        print("Epoch: " + str(epoch))
        if (epoch % each == 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] /= divide

        train(epoch)
        test(epoch)
