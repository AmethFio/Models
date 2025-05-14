import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import os
from DataSetting import MyDataset, DataSplitter
from TrainerTS import MyArgs

'''
Notes

1) Wi2Vi uses 56x3x3x29 CSI, while we use 30x3x3x100

2) Video frames are aligned with the first packets of CSI

3) Wi2Vi video FPS = 30 -> 6, CSI rate = 100Hz

4) Wi2Vi train:test = 95:5

5) Wi2Vi lr=2e-3 and lower; epoch=1000; batch size=32

6) Wi2Vi outputs 320x240 images

'''


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        """
        Interpolation layer
        :param size: (height, width)
        :param mode: interpolation mode
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class DropIn(nn.Module):
    def __init__(self, num_select):
        super(DropIn, self).__init__()
        self.num_select = num_select

    def forward(self, x):
        i = torch.randperm(x.shape[-1])[:self.num_select]
        return x[..., i]


class Wi2Vi(nn.Module):
    def __init__(self):
        super(Wi2Vi, self).__init__()

        # 56X29X18 (3x3xamp&phase)
        self.Dropin = DropIn(17)
        self.EncoderOriginal = nn.Sequential(
            # 56x17x18
            nn.Conv2d(18, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 56x15x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 26x7x128
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 12x3x256
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x1x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x1x512
        )

        self.Encoder = nn.Sequential(
            # 30x17x6
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 28x15x64
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 13x7x128
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            # 11x5x256
            nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x2x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(),
            # 5x2x512
        )

        self.Translator_A = nn.Sequential(
            # Please fill in the flattened output shape of Encoder.
            nn.Linear(5120, 972),
            nn.LeakyReLU()
        )

        self.Translator_B = nn.Sequential(
            # 36x27
            nn.ReflectionPad2d(1),
            # 38x29
            nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32x23x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 16x12x64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 8x6x128
        )

        self.Decoder = nn.Sequential(
            # 8x6x128
            # nn.ReflectionPad2d(1),
            # 10x8x128
            ResidualBlock(128, 128),
            # 8x6x128
            ResidualBlock(128, 128),
            # 8x6x128
            ResidualBlock(128, 128),
            # 8x6x128
            Interpolate(size=(12, 16)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            # 14x10x64
            Interpolate(size=(20, 28)),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            # 26x18x32
            Interpolate(size=(36, 52)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0),
            # 50x34x16
            Interpolate(size=(68, 100)),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0),
            # 98x66x8
            Interpolate(size=(132, 196)),
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0),
            # 194x130x4
            Interpolate(size=(260, 388)),
            nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=0),
            # 386x258x2
            nn.Conv2d(2, 1, kernel_size=5, stride=1, padding=0),
            nn.InstanceNorm2d(32),
            nn.Sigmoid()
            # 382x254x1
        )

    def forward(self, x):
        x = self.Dropin(x)
        x = self.Encoder(x)
        x = self.Translator_A(x.view(-1, 5120))
        x = self.Translator_B(x.view(-1, 1, 27, 36))
        x = self.Decoder(x)

        return x[..., 7:247, 31:351]

    def __str__(self):
        return 'Wi2Vi'


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=8, active_func=nn.Sigmoid()):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.active_func = active_func

        self.EnCNN = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=(3, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=(2, 2), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 1), padding=0),
            nn.LeakyReLU(inplace=True),
        )
        self.EnLSTM = nn.LSTM(512, self.latent_dim, 2, batch_first=True, dropout=0.1)
        self.DeFC = nn.Sequential(
            nn.Linear(self.latent_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )
        self.DeCNN = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            self.active_func
        )

    def __str__(self):
        return f"AutoEncoder{self.latent_dim}"

    def forward(self, x):
        x = torch.chunk(x.view(-1, 2, 90, 100), 2, dim=1)
        x1 = self.EnCNN(x[0])
        x2 = self.EnCNN(x[1])

        z = torch.cat([x1, x2], dim=1)
        z, (final_hidden_state, final_cell_state) = self.EnLSTM.forward(z.view(-1, 512, 8 * 42).transpose(1, 2))
        out = self.DeFC(z[:, -1, :])
        out = self.DeCNN(out.view(-1, 128, 4, 4))
        return out


class AutoEncoderNew(nn.Module):
    def __init__(self, latent_dim=8, active_func=nn.Sigmoid()):
        super(AutoEncoderNew, self).__init__()
        self.latent_dim = latent_dim
        self.active_func = active_func

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 128, 5, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.EnFC = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16),
            nn.ReLU()
        )

        self.DeFC = nn.Sequential(
            nn.Linear(16, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )
        self.DeCNN = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1),
            self.active_func
        )

    def __str__(self):
        return f"AutoEncoder{self.latent_dim}"

    def forward(self, x):
        x = self.cnn(x)
        out = self.EnFC(x.view(-1, 512 * 7 * 7))
        out = self.DeFC(out)
        out = self.DeCNN(out.view(-1, 128, 4, 4))
        return out


def timer(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print("\nTotal training time:", end - start, "sec")
        return result

    return wrapper


class CompTrainer:
    def __init__(self, model, args, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.model = model
        self.args = args

        self.train_loss = self.__gen_train_loss__()
        self.valid_loss = self.__gen_train_loss__()
        self.test_loss = self.__gen_test__()
        self.plot_terms = self.__plot_terms__()

        self.temp_loss = {}

    @staticmethod
    def __gen_train_loss__():
        train_loss = {'learning_rate': [],
                      'epochs': [0],
                      'LOSS': []}
        return train_loss

    @staticmethod
    def __gen_test__():
        test_loss = {'LOSS': [],
                     'PRED': [],
                     'GT': [],
                     'IND': []
                     }
        return test_loss

    @staticmethod
    def __plot_terms__():
        terms = {'loss': {'LOSS': 'Loss'},
                 'predict': ('GT', 'PRED', 'IND'),
                 'test': {'GT': 'Ground Truth',
                          'PRED': 'Estimated'
                          }
                 }
        return terms

    @staticmethod
    def __plot_settings__():
        plt.rcParams['figure.figsize'] = (20, 10)
        plt.rcParams["figure.titlesize"] = 35
        plt.rcParams['lines.markersize'] = 10
        plt.rcParams['axes.titlesize'] = 30
        plt.rcParams['axes.labelsize'] = 30
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

    @staticmethod
    def colors(arrays):
        arr = -np.log(arrays)
        norm = plt.Normalize(arr.min(), arr.max())
        map_vir = cm.get_cmap(name='viridis')
        c = map_vir(norm(arr))
        return c

    def logger(self):
        """
        Logs learning rate and number of epochs before training.
        :return:
        """
        objs = (self.train_loss, self.valid_loss)
        for obj in objs:

            # First round
            if not obj['learning_rate']:
                obj['learning_rate'].append(self.args.learning_rate)
                obj['epochs'].append(self.args.epochs)

            else:
                # Not changing learning rate
                if self.args.learning_rate == obj['learning_rate'][-1]:
                    obj['epochs'][-1] += self.args.epochs

                # Changing learning rate
                if self.args.learning_rate != obj['learning_rate'][-1]:
                    last_end = self.train_loss['epochs'][-1]
                    obj['learning_rate'].append(self.args.learning_rate)
                    obj['epochs'].append(last_end + self.args.epochs)

    def current_title(self):
        return f"e{self.train_loss['epochs'][-1]}"

    def calculate_loss(self, csi, img, i=None):
        mask = torch.where(img > 0, 1., 0.)
        output = self.model(csi)
        loss = self.args.criterion(output, mask)
        self.temp_loss = {'LOSS': loss}
        return {'GT': mask,
                'PRED': output,
                'IND': i}

    @timer
    def train(self, autosave=False, notion=''):
        self.logger()
        optimizer = self.args.optimizer([{'params': self.model.parameters()}],
                                        lr=self.args.learning_rate)
        LOSS_TERMS = self.plot_terms['loss'].keys()

        for epoch in range(self.args.epochs):

            # =====================train============================
            self.model.train()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}
            for idx, data in enumerate(self.train_loader, 0):
                csi = data['csi'].to(torch.float32).to(self.args.device)
                img = data['img'].to(torch.float32).to(self.args.device)
                optimizer.zero_grad()

                PREDS = self.calculate_loss(csi, img)
                self.temp_loss['LOSS'].backward()
                optimizer.step()
                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())

                if idx % (len(self.train_loader) // 5) == 0:
                    print(f"\rCompModel: epoch={epoch}/{self.args.epochs}, batch={idx}/{len(self.train_loader)},"
                          f"loss={self.temp_loss['LOSS'].item()}", end='')
            for key in LOSS_TERMS:
                self.train_loss[key].append(np.average(EPOCH_LOSS[key]))

            # =====================valid============================
            self.model.eval()
            EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

            for idx, data in enumerate(self.valid_loader, 0):
                csi = data['csi'].to(torch.float32).to(self.args.device)
                img = data['img'].to(torch.float32).to(self.args.device)
                with torch.no_grad():
                    PREDS = self.calculate_loss(csi, img)
                for key in LOSS_TERMS:
                    EPOCH_LOSS[key].append(self.temp_loss[key].item())
            for key in LOSS_TERMS:
                self.valid_loss[key].append(np.average(EPOCH_LOSS[key]))

        if autosave:
            save_path = f'../saved/{notion}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.model.state_dict(),
                       f"{save_path}{notion}_{self.model}_{self.current_title()}.pth")

    def test(self, mode='test'):
        self.test_loss = self.__gen_test__()
        self.model.eval()
        LOSS_TERMS = self.plot_terms['loss'].keys()
        PLOT_TERMS = self.plot_terms['predict']
        EPOCH_LOSS = {key: [] for key in LOSS_TERMS}

        if mode == 'test':
            loader = self.test_loader
        elif mode == 'train':
            loader = self.train_loader

        for idx, data in enumerate(loader, 0):
            csi = data['csi'].to(torch.float32).to(self.args.device)
            img = data['img'].to(torch.float32).to(self.args.device)
            ind = data['ind']
            with torch.no_grad():
                for sample in range(loader.batch_size):
                    ind_ = ind[sample]
                    csi_ = csi[sample][np.newaxis, ...]
                    img_ = img[sample][np.newaxis, ...]
                    PREDS = self.calculate_loss(csi_, img_, ind_)

                    for key in LOSS_TERMS:
                        EPOCH_LOSS[key].append(self.temp_loss[key].item())

                    for key in PLOT_TERMS:
                        self.test_loss[key].append(PREDS[key].cpu().detach().numpy().squeeze())
                    for key in LOSS_TERMS:
                        self.test_loss[key] = EPOCH_LOSS[key]

            if idx % (len(loader)//5) == 0:
                print(f"\rCompModel: test={idx}/{len(loader)}, loss={self.temp_loss['LOSS'].item()}", end='')

        for key in LOSS_TERMS:
            EPOCH_LOSS[key] = np.average(EPOCH_LOSS[key])
        print(f"\nTest finished. Average loss={EPOCH_LOSS}")

    def plot_train_loss(self, double_y=False, autosave=False, notion=''):
        self.__plot_settings__()

        PLOT_ITEMS = self.plot_terms['loss']
        stage_color = self.colors(self.train_loss['learning_rate'])
        line_color = ['b', 'orange']
        epoch = self.train_loss['epochs'][-1]
        save_path = f'../saved/{notion}/'

        title = f"{self.model} Training Status @ep{epoch}"

        filename = f"{save_path}{notion}_{self.model}_train_{self.current_title()}.jpg"

        # Training & Validation Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title)
        if len(PLOT_ITEMS.keys()) == 1:
            axes = [plt.gca()]
        elif len(PLOT_ITEMS.keys()) > 3:
            axes = fig.subplots(2, np.ceil(len(PLOT_ITEMS.keys())/2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(PLOT_ITEMS.keys()))
            axes = axes.flatten()

        for i, loss in enumerate(PLOT_ITEMS.keys()):
            for j, learning_rate in enumerate(self.train_loss['learning_rate']):
                axes[i].axvline(self.train_loss['epochs'][j],
                                linestyle='--',
                                color=stage_color[j],
                                label=f'lr={learning_rate}')

            axes[i].plot(list(range(len(self.valid_loss[loss]))),
                         self.valid_loss[loss],
                         line_color[1], label='Valid')
            if double_y:
                ax_r = axes[i].twinx()
            else:
                ax_r = axes[i]
            ax_r.plot(list(range(len(self.train_loss[loss]))),
                      self.train_loss[loss],
                      line_color[0], label='Train')
            axes[i].set_title(PLOT_ITEMS[loss])
            axes[i].set_xlabel('#Epoch')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            axes[i].legend()

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(filename)
        plt.show()

    def plot_test(self, select_ind=None, select_num=8, autosave=False, notion=''):
        # self.__plot_settings__()
        PLOT_ITEMS = self.plot_terms['test']
        LOSS_ITEMS = self.plot_terms['loss']
        epoch = self.train_loss['epochs'][-1]
        save_path = f'../saved/{notion}/'

        title = {'PRED': f"{self.model} Test Predicts @ep{epoch}",
                 'LOSS': f"{self.model} Test Loss @ep{epoch}"}
        filename = {'PRED': f"{save_path}{notion}_{self.model}_predict_{self.current_title()}.jpg",
                    'LOSS': f"{save_path}{notion}_{self.model}_test_{self.current_title()}.jpg"}

        if select_ind:
            inds = select_ind
        else:
            inds = np.random.choice(list(range(len(self.test_loss['IND']))), select_num, replace=False)
        inds = np.sort(inds)
        samples = np.array(self.test_loss['IND'])[inds]

        # Depth Images
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title['PRED'])
        subfigs = fig.subfigures(nrows=len(PLOT_ITEMS.keys()), ncols=1)

        for i, item in enumerate(PLOT_ITEMS.keys()):
            subfigs[i].suptitle(PLOT_ITEMS[item])
            axes = subfigs[i].subplots(nrows=1, ncols=select_num)
            for j in range(len(axes)):
                img = axes[j].imshow(self.test_loss[item][inds[j]], vmin=0, vmax=1)
                axes[j].axis('off')
                axes[j].set_title(f"#{samples[j]}")
            subfigs[i].colorbar(img, ax=axes, shrink=0.8)

        if autosave:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(filename['PRED'])
        plt.show()

        # Test Loss
        fig = plt.figure(constrained_layout=True)
        fig.suptitle(title['LOSS'])
        if len(LOSS_ITEMS.keys()) == 1:
            axes = [plt.gca()]
        elif len(LOSS_ITEMS.keys()) > 3:
            axes = fig.subplots(2, np.ceil(len(LOSS_ITEMS.keys())/2).astype(int))
            axes = axes.flatten()
        else:
            axes = fig.subplots(1, len(LOSS_ITEMS.keys()))
            axes = axes.flatten()

        for i, item in enumerate(LOSS_ITEMS.keys()):
            axes[i].scatter(list(range(len(self.test_loss[item]))),
                            self.test_loss[item], alpha=0.6)
            axes[i].set_title(LOSS_ITEMS[item])
            axes[i].set_xlabel('#Sample')
            axes[i].set_ylabel('Loss')
            axes[i].grid()
            for j in range(select_num):
                axes[i].scatter(inds[j], self.test_loss[item][inds[j]],
                                c='magenta', marker=(5, 1), linewidths=4)

        if autosave:
            plt.savefig(filename['LOSS'])
        plt.show()

    def scheduler(self, turns=10, lr_decay=False, decay_rate=0.4, test_mode='train', autosave=False, notion=''):

        for i in range(turns):
            self.train()
            self.test(mode=test_mode)
            self.plot_test(autosave=autosave, notion=notion)
            self.plot_train_loss(autosave=autosave, notion=notion)
            if lr_decay:
                self.args.learning_rate *= decay_rate

        print("\nSchedule Completed!")

    def save_all_params(self, notion=''):
        save_path = f'../saved/{notion}/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.model.state_dict(),
                   f"{save_path}{notion}_{self.model}_{self.current_title()}.pth")


if __name__ == "__main__":
    #m1 = Wi2Vi()
    #summary(m1, input_size=(6, 30, 30))
    m2 = AutoEncoderNew()
    summary(m2, input_size=(6, 30, 30))

