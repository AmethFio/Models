import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from TrainerTS import timer, MyDataset, split_loader, MyArgs, TrainerTeacherStudent, bn, Interpolate


# ------------------------------------- #
# Model v03b1
# Added interpolating decoder
# Adaptive to MNIST

# ImageEncoder: in = 128 * 128, out = 1 * latent_dim
# ImageDecoder: in = 1 * latent_dim, out = 128 * 128
# CSIEncoder: in = 2 * 90 * 100, out = 1 * 256


class ImageEncoder(nn.Module):
    def __init__(self, bottleneck='fc', batchnorm=False, latent_dim=8):
        super(ImageEncoder, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 128 * 128 * 1
            # Out = 64 * 64 * 16
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 64 * 64 * 16
            # Out = 32 * 32 * 32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 32 * 32 * 32
            # Out = 16 * 16 * 64
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 16 * 16 * 64
            # Out = 8 * 8 * 128
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
            # In = 8 * 8 * 128
            # Out = 4 * 4 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=(4, 4), stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.latent_dim),
            # nn.Sigmoid()
        )

    def __str__(self):
        return 'ImgEnV03b1' + self.bottleneck.capitalize()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.bottleneck == 'fc':
            x = self.fclayers(x.view(-1, 4 * 4 * 256))
        elif self.bottleneck == 'gap':
            x = self.gap(x)
            x = nn.Sigmoid(x)

        return x.view(-1, self.latent_dim)


class ImageDecoder(nn.Module):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 1 * 1 * 256
            # Out = 4 * 4 * 128
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 4 * 4 * 128
            # Out = 8 * 8 * 64
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 8 * 8 * 64
            # Out = 16 * 16 * 32
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 16 * 16 * 32
            # Out = 32 * 32 * 16
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            # In = 32 * 32 * 16
            # Out = 64 * 64 * 8
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 64 * 64 * 8
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeV03b1' + self.fc

    def forward(self, x):

        x = self.fclayers(x.view(-1, self.latent_dim))
        x = self.layer1(x.view(-1, 256, 1, 1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x.view(-1, 1, 128, 128)


class ImageDecoderInterp(ImageDecoder):
    def __init__(self, batchnorm=False, latent_dim=8, active_func=nn.Sigmoid()):
        super(ImageDecoderInterp, self).__init__()

        self.latent_dim = latent_dim
        self.active_func = active_func

        self.fclayers = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(16, 16)),
            # In = 4 * 4 * 32
            # Out = 16 * 16 * 16
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            bn(8, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(64, 64)),
            # In = 16 * 16 * 16
            # Out = 64 * 64 * 8
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            bn(4, batchnorm),
            nn.LeakyReLU(inplace=True),
            Interpolate(size=(128, 128)),
            # In = 64 * 64 * 8
            # Out = 128 * 128 * 4
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
            bn(1, batchnorm),
            self.active_func,
            # In = 128 * 128 * 4
            # Out = 128 * 128 * 1
        )

    def __str__(self):
        return 'ImgDeV03b1Interp' + self.fc

    def forward(self, x):

        z = self.fclayers(x.view(-1, self.latent_dim))
        z = self.layer1(z.view(-1, 32, 4, 4))
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)

        return z.view(-1, 1, 128, 128)


class CsiEncoder(nn.Module):
    def __init__(self, bottleneck='last', batchnorm=False, latent_dim=8):
        super(CsiEncoder, self).__init__()

        self.bottleneck = bottleneck
        self.latent_dim = latent_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            bn(16, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 90 * 100 * 1
            # Out = 30 * 98 * 16
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=0),
            bn(32, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 30 * 98 * 16
            # Out = 14 * 48 * 32
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=0),
            bn(64, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 14 * 48 * 32
            # Out = 12 * 46 * 64
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            bn(128, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 12 * 46 * 64
            # Out = 10 * 44 * 128
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            bn(256, batchnorm),
            nn.LeakyReLU(inplace=True),
            # No Padding
            # No Pooling
            # In = 10 * 44 * 128
            # Out = 8 * 42 * 256
        )

        self.gap = nn.Sequential(
            nn.AvgPool1d(kernel_size=8 * 42, stride=1, padding=0)
        )

        self.fclayers = nn.Sequential(
            nn.Linear(256 * 8 * 42, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            # nn.ReLU()
        )

        self.lstm = nn.Sequential(
            nn.LSTM(512, 2 * self.latent_dim, 2, batch_first=True, dropout=0.1)
        )

    def __str__(self):
        return 'CsiEnV03b1' + self.bottleneck.capitalize()

    def forward(self, x):
        x = torch.chunk(x.view(-1, 2, 90, 100), 2, dim=1)
        x1 = self.layer1(x[0])
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)
        x1 = self.layer5(x1)

        x2 = self.layer1(x[1])
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        x2 = self.layer5(x2)

        out = torch.cat([x1, x2], dim=1)
        out, (final_hidden_state, final_cell_state) = self.lstm.forward(out.view(-1, 512, 8 * 42).transpose(1, 2))

        if self.bottleneck == 'full_fc':
            out = self.fclayers(out.view(-1, 256 * 8 * 42))

        elif self.bottleneck == 'full_gap':
            out = self.gap(out.transpose(1, 2))

        elif self.bottleneck == 'last':
            out = out[:, -1, :]

        return out


if __name__ == "__main__":
    m1 = ImageEncoder(batchnorm=False, latent_dim=8)
    summary(m1, input_size=(1, 128, 128))