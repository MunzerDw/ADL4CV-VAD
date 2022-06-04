import torch
import torch.nn as nn

class ThreeDEPNDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_features = 32

        self.bottleneck = nn.Sequential(
            nn.Linear(self.num_features*8, self.num_features*8),
            nn.ReLU(),
            nn.Linear(self.num_features*8, self.num_features*16),
            nn.ReLU(),
        ) # batch_size x 512

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features*16, out_channels=self.num_features*8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm3d(self.num_features*8),
            nn.ReLU()
        ) # batch_size x 256 x 4 x 4 x 4

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features * 8, out_channels=self.num_features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features*4),
            nn.ReLU()
        ) # batch_size x 128 x 8 x 8 x 8

        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features * 4, out_channels=self.num_features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(self.num_features*2),
            nn.ReLU()
        ) # batch_size x 64 x 16 x 16 x 16

        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=self.num_features * 2, out_channels=1, kernel_size=4, stride=2, padding=1)
        ) # batch_size x 1 x 32 x 32 x 32

    def forward(self, x):
        # bottleneck layers
        x = self.bottleneck(x)
        # reshape for ConvTranspose3d
        x = x.view(x.shape[0], x.shape[1], 1, 1, 1)

        # Decode
        x_d1 = self.decoder1(x)
        x_d2 = self.decoder2(x_d1)
        x_d3 = self.decoder3(x_d2)
        x = self.decoder4(x_d3)
        x = torch.squeeze(x, dim=1)
        x = torch.log(torch.abs(x)+1)
        
        return x
