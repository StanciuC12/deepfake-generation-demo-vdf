
# !git clone https://github.com/StanciuC12/deepfake-generation-demo-vdf.git
# %cd deepfake-generation-demo-vdf

#from LandMarkDetector import LandMarkDetector
import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(128, 256, 2, padding=1, stride=2)
        self.conv3 = nn.Conv2d(256, 512, 2, padding=1, stride=2)
        self.conv4 = nn.Conv2d(512, 1024, 2, padding=1, stride=2)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x)) #compressed representation

        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv6 = nn.ConvTranspose2d(64, 3, 2, stride=2)

    def forward(self, x):

        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.relu(self.t_conv5(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv6(x))

        return x



# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#
#         ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.t_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.t_conv5 = nn.ConvTranspose2d(64, 64, 2, stride=2)
#         self.t_conv6 = nn.ConvTranspose2d(64, 3, 2, stride=2)
#
#     def forward(self, x):
#
#         ## decode ##
#         # add transpose conv layers, with relu activation function
#         x = F.relu(self.t_conv1(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv3(x))
#         x = F.relu(self.t_conv4(x))
#         x = F.relu(self.t_conv5(x))
#         # output layer (with sigmoid for scaling from 0 to 1)
#         x = F.sigmoid(self.t_conv6(x))
#
#         return x




if __name__ == '__main__':
    ######################
    #######TRAINING#######
    ######################


    e = Encoder()
    e.to('cuda')
    d1 = Decoder()
    d1.to('cuda')
    d2 = Decoder()
    d2.to('cuda')
    input = torch.rand(3, 256, 256).to('cuda')


    #parameters###############################################
    n_epochs = 20
    lr_e = 1e-4
    lr_d = 5e-3
    criterion = nn.BCELoss()
    optimizer_e = torch.optim.Adam(e.parameters(), lr=lr_e)
    optimizer_d1 = torch.optim.Adam(d1.parameters(), lr=lr_d)
    optimizer_d2 = torch.optim.Adam(d2.parameters(), lr=lr_d)
    ##########################################################

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        for i in range(10):
            # _ stands in for labels, here
            # no need to flatten images
            images = torch.rand(16, 3, 256, 256).to('cuda')
            # clear the gradients of all optimized variables
            optimizer_e.zero_grad()
            optimizer_d1.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            out = d1(e(images))
            # calculate the loss
            loss = criterion(out, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer_e.step()
            optimizer_d1.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)

        # print avg training statistics
        train_loss = train_loss/160
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
            ))
