import torch
import torch.nn as nn
import torch.nn.functional as F
from Structure.Model import batchnorm_layer, reparameterize, GEGLU_proj

class CSIEncoder3V(nn.Module):
    name = 'csien'
    
    def __init__(self, batchnorm='identity', latent_dim=16, lstm_step_length=512 * 7, lstm_steps=75, *args, **kwargs):
        super(CSIEncoder3V, self).__init__(*args, **kwargs)

        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        self.lstm_step_length = lstm_step_length

        self.lstm_steps = lstm_steps
        self.csi_feature_length = 128
        self.pd_feature_length = 128
        self.feature_length = 1536
        self.pd_length = 62

        # 6 * 30 * 100
        # 128 * 28 * 98
        # 256 * 14 * 49
        # 512 * 7 * 25

        self.cnn = nn.Sequential(
            nn.Conv2d(6, 128, 5, 1, 1),
            batchnorm_layer(128, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            batchnorm_layer(256, self.batchnorm),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            batchnorm_layer(512, self.batchnorm),
            nn.LeakyReLU(inplace=True)
        )

        self.lstm = nn.LSTM(self.lstm_step_length, self.csi_feature_length, 2, batch_first=True, dropout=0.1)
        
        self.fc_feature = nn.Sequential(
            nn.Linear(self.csi_feature_length + self.pd_feature_length, 
                      self.feature_length),
            nn.ReLU()
        )
        
        self.fc_pd = nn.Sequential(
            nn.Linear(self.pd_length, self.pd_feature_length),
            nn.ReLU()
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(self.feature_length, self.latent_dim)
            # nn.ReLU()
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(self.feature_length, self.latent_dim)
            # nn.ReLU()
        )

    def __str__(self):
        return f"CSIEN_tripleV"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        features, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512*7, 75).transpose(1, 2))
        # 256-dim output
        out1 = torch.cat((features[:, 24, :].view(-1, 128), fea_pd.view(-1, 128)), -1)
        out2 = torch.cat((features[:, 49, :].view(-1, 128), fea_pd.view(-1, 128)), -1)
        out3 = torch.cat((features[:, -1, :].view(-1, 128), fea_pd.view(-1, 128)), -1)
        out1 = self.fc_feature(out1)
        out2 = self.fc_feature(out2)
        out3 = self.fc_feature(out3)
        
        mu1 = self.fc_mu(out1)
        logvar1 = self.fc_logvar(out1)
        z1 = reparameterize(mu1, logvar1)
        
        mu2 = self.fc_mu(out2)
        logvar2 = self.fc_logvar(out2)
        z2 = reparameterize(mu2, logvar2)
        
        mu3 = self.fc_mu(out3)
        logvar3 = self.fc_logvar(out3)
        z3 = reparameterize(mu3, logvar3)
        # return [out1, out2, out3], [z1, z2, z3], [mu1, mu2, mu3], [logvar1, logvar2, logvar3]
        return [out1, out2, out3], [z1, z2, z3], [mu1, mu2, mu3], [logvar1, logvar2, logvar3]


class CSIEncoderAPool(CSIEncoder3V):
    name = 'csien'
    
    def __init__(self, *args, **kwargs):
        super(CSIEncoderAPool, self).__init__(*args, **kwargs)
        self.fc_feature = GEGLU_proj(self.csi_feature_length + self.pd_feature_length, 
                      self.feature_length)

    def __str__(self):
        return f"CSIENAPool"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        features, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, self.lstm_step_length, self.lstm_steps).transpose(1, 2))
        # 256-dim output
        features = features.mean(dim=1)
        out = torch.cat((features.view(-1, self.csi_feature_length), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(out)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)
        return out, z, mu, logvar


class CSIEncoderAttn(CSIEncoder3V):
    name = 'csien'

    def __init__(self, *args, **kwargs):
        super(CSIEncoderAttn, self).__init__(*args, **kwargs)
        self.attn_vector = nn.Parameter(torch.randn(128))  # Learnable attention vector
        self.fc_feature = GEGLU_proj(self.csi_feature_length + self.pd_feature_length, 
                      self.feature_length)

    def __str__(self):
        return f"CSIENAttn"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        features, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512*7, 75).transpose(1, 2))

        # Attention weights
        attn_scores = torch.matmul(features, self.attn_vector)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, seq_len]

        # Apply attention weights
        context = torch.sum(features * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]

        out = torch.cat((context.view(-1, self.csi_feature_length), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(out)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class CSIEncoderHPool(CSIEncoder3V):
    name = 'csien'

    def __init__(self, *args, **kwargs):
        super(CSIEncoderHPool, self).__init__(*args, **kwargs)

        # self.fc_feature = nn.Sequential(
        #     nn.Linear(self.csi_feature_length * 3 + self.pd_feature_length, 
        #               self.feature_length),
        #     nn.ReLU()
        # )
        
        self.fc_feature = GEGLU_proj(self.csi_feature_length * 3 + self.pd_feature_length, 
                       self.feature_length)

    def __str__(self):
        return f"CSIENHP"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512*7, 75).transpose(1, 2))

        chunks = torch.chunk(lstm_out, 3, dim=1)
        summaries = [chunk.mean(dim=1) for chunk in chunks]
        features = torch.cat(summaries, dim=1)

        out = torch.cat((features.view(-1, self.csi_feature_length * 3), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(out)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar


class CSIEncoderConcat(CSIEncoder3V):
    name = 'csien'

    def __init__(self, *args, **kwargs):
        super(CSIEncoderConcat, self).__init__(*args, **kwargs)

        self.fc_feature = nn.Sequential(
            nn.Linear(self.csi_feature_length * 3 + self.pd_feature_length, 
                      self.feature_length),
            nn.ReLU()
        )
        # self.fc_feature = GEGLU_proj(self.csi_feature_length * 3 + self.pd_feature_length, 
        #                self.feature_length)

    def __str__(self):
        return f"CSIENCon"

    def forward(self, csi, pd):
        fea_csi = self.cnn(csi)
        fea_pd = self.fc_pd(pd)
        lstm_out, (final_hidden_state, final_cell_state) = self.lstm.forward(
            fea_csi.view(-1, 512*7, 75).transpose(1, 2))

        features = torch.cat((lstm_out[:, 24, :], lstm_out[:, 49, :],lstm_out[:, -1, :]), dim=1)

        out = torch.cat((features.view(-1, self.csi_feature_length * 3), fea_pd.view(-1, self.pd_feature_length)), -1)
        out = self.fc_feature(out)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        z = reparameterize(mu, logvar)

        return out, z, mu, logvar