import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FFLayer(nn.Module):
    """
    Feed-forward layer with layer-norm, dropout, activation
    """
    def __init__(self, input_size, output_size, layer_norm, dropout, activation):
        super(FFLayer, self).__init__()

        self.layer = nn.Linear(input_size, output_size)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(input_size)
        else:
            self.layer_norm = lambda x: x
        
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

        if activation == 'relu':
            self.activation = lambda x: F.relu(x)
        elif activation == 'linear':
            self.activation = lambda x: x
        else:
            raise NotImplementedError

    def forward(self, inputs):
        normed = self.layer_norm(inputs)
        output = self.layer(normed)
        dropped = self.dropout(output)
        return self.activation(dropped)

    def freeze_weights(self):
        for param in self.layer.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.layer.parameters():
            param.requires_grad = True

class MaskedDecoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, output_size, layer_norm, dropout,
                activation):
        super(MaskedDecoder, self).__init__()

        self.latent_size = latent_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.activation = activation

        # nn layers
        sizes = [self.latent_size] + self.hidden_sizes + [output_size]
        self.hiddens = []
        for i, s in enumerate(sizes[:-2]):
            self.hiddens.append(FFLayer(s, sizes[i+1], self.layer_norm, self.dropout, self.activation))
        # last layer has no activation
        self.hiddens.append(FFLayer(sizes[-2], sizes[-1], self.layer_norm, self.dropout, 'linear'))

        self.hiddens = nn.ModuleList(self.hiddens)

    def init_latents(self, n_points, device, mode):
        # TODO: return different latent types depending on option
        if mode == 'zeros':
            return torch.zeros(n_points, self.latent_size, dtype=torch.float,
                        requires_grad=True, device=device)
        elif mode == 'normal':
            stdev = 0.001
            print("Initializing from normal distribution with stdev {}".format(stdev), file=sys.stderr)

            return torch.tensor(np.random.normal(0, stdev, size=(n_points, self.latent_size)), dtype=torch.float,
                requires_grad=True, device=device)

            #return torch.randn(n_points, self.latent_size, dtype=torch.float,
            #            requires_grad=True, device=device) * stdev
        else:
            raise NotImplementedError
        # return torch.randn(n_points, self.latent_size, requires_grad=True, device=device)

    def forward(self, latents):
        # x = F.normalize(latents)
        x = latents

        for layer in self.hiddens:
            x = layer(x)
        return x

    def freeze_hiddens(self):
        for h in self.hiddens:
            h.freeze_weights()

    def unfreeze_hiddens(self):
        for h in self.hiddens:
            h.unfreeze_weights()

class MaskedVAD(MaskedDecoder):
    def __init__(self, *args, log_var=0.):
        super(MaskedVAD, self).__init__(*args)

        print("log var: {}".format(log_var))
        log_var_param = nn.Parameter(data=torch.tensor(log_var), requires_grad=False)
        self.register_parameter('log_var', log_var_param)
        self.std = np.exp(log_var / 2)
        print("std: {}".format(self.std))

        self.verbose = False

    def set_verbose(self, verb):
        self.verbose = verb

    def forward(self, latents):
        # sample with unit variance
        # eps = torch.randn_like(latents, device=latents.device)
        # x = latents + torch.exp(self.log_var / 2) * eps
        
        x = latents + torch.normal(torch.zeros_like(latents), std=self.std)

        # x = torch.randn_like(latents) + latents

        if self.verbose:
            print(latents[:10, :10])
            print(x[:10, :10])
            print(torch.exp(self.log_var / 2))
            print(eps[:10, :10])

        for layer in self.hiddens:
            x = layer(x)

        return x

class MaskedVAD_free(MaskedDecoder):
    def __init__(self, *args):
        super(MaskedVAD_free, self).__init__(*args)

        self.verbose = False

    def set_verbose(self, verb):
        self.verbose = verb

    def forward(self, latents, latent_log_var):
        # sample with unit variance
        # eps = torch.randn_like(latents, device=latents.device)
        # x = latents + torch.exp(self.log_var / 2) * eps
        
        x = latents + torch.randn_like(latents) * torch.exp(0.5 * latent_log_var)

        if self.verbose:
            print(latents[:10, :10])
            print(x[:10, :10])
            print(torch.exp(self.log_var / 2))
            print(eps[:10, :10])

        for layer in self.hiddens:
            x = layer(x)

        return x

class MaskedVAD2(MaskedDecoder):
    """
    Specifically for SFM tasks, with intermediate 3D output with 2D projection
    """
    def __init__(self, *args, log_var=0.):
        super(MaskedVAD2, self).__init__(*args)

        print("log var: {}".format(log_var))
        log_var_param = nn.Parameter(data=torch.tensor(log_var), requires_grad=False)
        self.register_parameter('log_var', log_var_param)
        self.std = np.exp(log_var / 2)
        print("std: {}".format(self.std))

        self.verbose = False
        
        # transformation layers
        # for now: directly output the 4x3 transformation matrix

        self.transform_layer = FFLayer(self.latent_size, 4 * 3, self.layer_norm, self.dropout, 'linear')

    def set_verbose(self, verb):
        self.verbose = verb

    def forward(self, latents):
        # sample with unit variance
        # eps = torch.randn_like(latents, device=latents.device)
        # x = latents + torch.exp(self.log_var / 2) * eps
        
        x = latents + torch.normal(torch.zeros_like(latents), std=self.std)

        # x = torch.randn_like(latents) + latents

        transform_mat = self.transform_layer(x).view(-1, 4, 3)

        if self.verbose:
            print(latents[:10, :10])
            print(x[:10, :10])
            print(torch.exp(self.log_var / 2))
            print(eps[:10, :10])

        for layer in self.hiddens:
            x = layer(x)

        x = x.view(-1, 84, 3)
        additional = torch.ones(x.size()[:-1] + (1,), device=x.device)

        coords = torch.cat([x, additional], dim=-1)
        transformed = torch.matmul(coords, transform_mat)

        return transformed[:, :, :2].contiguous().view(-1, 84 * 2), transform_mat



