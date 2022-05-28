import torch.nn as nn
import torch


class DeepSDFDecoder(nn.Module):

    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2
        self.layer1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(259, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 253), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 1), name='weight')
        )


    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        x = self.layer1(x_in)
        x = self.layer2(torch.cat([x,x_in], dim=1))
        return x




class DeepSDFVAD(nn.Module):
    def __init__(self, latent_size):
        """
        :param latent_size: latent code vector length
        """
        super().__init__()
        dropout_prob = 0.2
        self.layer1 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(259, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 253), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 512), name='weight'),
            nn.Dropout(dropout_prob),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(512, 1), name='weight')
        )

    def forward(self, x_in):
        """
        :param x_in: B x (latent_size + 3) tensor
        :return: B x 1 tensor
        """
        #x_code = x_in[:, 0: 256]
        #x_coord = x_in[:, 256: 259]
        #x_vad = x_code + torch.randn_like(x_code) * torch.exp(0.5 * log_var)
        #x_vad = torch.cat([x_vad, x_coord], dim=1)
        #x = self.layer1(x_vad)
        #x = self.layer2(torch.cat([x, x_vad], dim=1))
        x = self.layer1(x_in)
        x = self.layer2(torch.cat([x, x_in], dim=1))
        return x

