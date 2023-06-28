# imports
from LandMarkDetector import LandMarkDetector
from DataLoader import DataLoader
from models import Encoder, Decoder
import os
# Train
import torch.nn as nn
import torch.nn.functional as F
import torch


# Preprocess data
detector = LandMarkDetector()
video_adrs = [os.path.join('videos', 'id31_0005.mp4'), os.path.join('videos', 'id19_0006.mp4')]
os.mkdir('out')
for video_adr in video_adrs:
  detector.video2croppedImages(video_path=video_adr, name_prefix=video_adr.split('/')[-1].split('.')[0])



# initialize parameters, data loader, models
device = 'cuda'
e = Encoder()
e.to(device)
d1 = Decoder()
d1.to(device)
d2 = Decoder()
d2.to(device)

# parameters###############################################
n_epochs = 20
lr_e = 1e-4
lr_d = 5e-3
batch_size = 4
criterion = nn.BCELoss()
optimizer_e = torch.optim.Adam(e.parameters(), lr=lr_e)
optimizer_d1 = torch.optim.Adam(d1.parameters(), lr=lr_d)
optimizer_d2 = torch.optim.Adam(d2.parameters(), lr=lr_d)
##########################################################

data_loader = DataLoader(classes=['id19_', 'id31_'], batch_size=batch_size)
nr_batches = data_loader.min_batches_each_class
for epoch in range(1, n_epochs + 1):
    # monitor training loss
    train_loss = 0.0
    data_loader.shuffle_data()
    ###################
    # train the model #
    ###################
    for i in range(nr_batches):
        print(f'{str(i)}/{str(nr_batches)}')
        images_c0 = data_loader.get_data(class_nr=0, batch_nr=i).to(device)

        # clear the gradients of all optimized variables
        optimizer_e.zero_grad()
        optimizer_d1.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        out1 = d1(e(images_c0))

        # calculate the loss
        loss1 = criterion(out1, images_c0)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss1.backward()
        # perform a single optimization step (parameter update)
        optimizer_e.step()
        optimizer_d1.step()
        # update running training loss
        train_loss += loss1.item() * images_c0.size(0)

        images_c1 = data_loader.get_data(class_nr=1, batch_nr=i).to(device)
        optimizer_e.zero_grad()
        optimizer_d2.zero_grad()
        out2 = d2(e(images_c1))
        loss2 = criterion(out2, images_c1)
        loss2.backward()
        optimizer_e.step()
        optimizer_d2.step()
        train_loss += loss2.item() * images_c1.size(0)

    # print avg training statistics
    train_loss = train_loss / nr_batches
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch,
        train_loss
    ))

    # Saving models
    torch.save(e.state_dict(), f'encoder_e{str(epoch)}.pkl')
    torch.save(d1.state_dict(), f'decoder1_e{str(epoch)}.pkl')
    torch.save(d2.state_dict(), f'decoder2_e{str(epoch)}.pkl')


# Evaluation
