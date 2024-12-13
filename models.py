import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaSequential)
# from torchmeta.modules.utils import self.get_subdict


# Code adapted from Sitzmann et al 2020
# https://github.com/vsitzmann/siren

def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class FCBlock(MetaModule):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, outermost_linear=False,
                 nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(MetaSequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(MetaSequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(MetaSequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = MetaSequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords, params=None, **kwargs):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # output = self.net(coords, params=self.self.get_subdict(params, 'net'))
        output = self.net(coords, params=self.get_subdict(params, 'net'))
        return output

    def forward_with_activations(self, coords, params=None, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.'''
        if params is None:
            params = OrderedDict(self.named_parameters())

        activations = OrderedDict()

        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            subdict = self.self.get_subdict(params, 'net.%d' % i)
            for j, sublayer in enumerate(layer):
                if isinstance(sublayer, BatchLinear):
                    x = sublayer(x, params=self.self.get_subdict(subdict, '%d' % j))
                else:
                    x = sublayer(x)

                if retain_grad:
                    x.retain_grad()
                activations['_'.join((str(sublayer.__class__), "%d" % i))] = x
        return activations


class SetEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features, nonlinearity='relu'):
        super().__init__()

        assert nonlinearity in ['relu'], 'Unknown nonlinearity type'

        if nonlinearity == 'relu':
            nl = nn.ReLU(inplace=True)
            weight_init = init_weights_normal

        self.net = [nn.Linear(in_features, hidden_features), nl]
        self.net.extend([nn.Sequential(nn.Linear(hidden_features, hidden_features), nl)
                         for _ in range(num_hidden_layers)])
        self.net.extend([nn.Linear(hidden_features, out_features), nl])
        self.net = nn.Sequential(*self.net)

        self.net.apply(weight_init)

    def forward(self, context_x, context_y, **kwargs):  # ctxt_mask=None,
        input = torch.cat((context_x, context_y), dim=-1)
        embeddings = self.net(input)

        return embeddings.mean(dim=-2)


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''
        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                         num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                         outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class SingleBVPNet(MetaModule):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='relu', in_features=2, mode='mlp', hidden_features=256, num_hidden_layers=3,
                 **kwargs):
        super().__init__()
        self.mode = mode

        self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        # print(self)

    def forward(self, model_input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        # Enables us to compute gradients w.r.t. coordinates
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        coords = coords_org

        output = self.net(coords, self.get_subdict(params, 'net'))
        return {'model_in': coords_org, 'model_out': output}

    def forward_with_activations(self, model_input):
        '''Returns not only model output, but also intermediate activations.'''
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        activations = self.net.forward_with_activations(coords)
        return {'model_in': coords, 'model_out': activations.popitem(), 'activations': activations}

    # Initialization schemes


def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1 / in_features_main_net, 1 / in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)


class HyperBRDF(nn.Module):

    def __init__(self, in_features, out_features, encoder_nl='relu'):
        super().__init__()

        latent_dim = 40
        self.hypo_net = SingleBVPNet(out_features=out_features, hidden_features=60, type='relu',
                                     in_features=6)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=128,
                                      hypo_module=self.hypo_net)
        self.set_encoder = SetEncoder(in_features=9, out_features=latent_dim, num_hidden_layers=2,
                                      hidden_features=128, nonlinearity=encoder_nl)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def forward(self, model_input):
        amps, coords = model_input['amps'], model_input['coords']

        # Embed coords and amplitudes
        embedding = self.set_encoder(coords, amps)

        # Take the embeddings from the set encoder and pass them through the hypernetwork (decoder).
        # Outputs the hypo_nets weights
        hypo_params = self.hyper_net(embedding)

        # From model_input it will take the coordenites and use them to obtain the amplitudes from the hypo_net
        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}
        
        

    
if __name__ == '__main__':
    import argparse
    import sys
    from torch.utils.data import DataLoader
    from data_processing import MerlDataset
    import time
    
    # Create test arguments
    test_args = [
        '--destdir', 'results/merl',
        '--binary', 'data',
        '--dataset', 'MERL',
        '--kl_weight', '0.1',
        '--fw_weight', '0.1',
        '--epochs', '100',
        '--lr', '5e-5',
        '--keepon', 'False'
    ]
    
    # Parse arguments using the test values
    parser = argparse.ArgumentParser('HyperBRDF Training')
    parser.add_argument('--destdir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--binary', type=str, required=True,
                        help='Dataset path')
    parser.add_argument('--dataset', choices=['MERL', 'EPFL'], default='MERL',
                        help='Dataset type (MERL or EPFL)')
    parser.add_argument('--kl_weight', type=float, default=0.1,
                        help='Weight for KL divergence loss')
    parser.add_argument('--fw_weight', type=float, default=0.1,
                        help='Weight for forward loss')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--keepon', type=bool, default=False,
                        help='Continue training from loaded checkpoint')
    
    args = parser.parse_args(test_args)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating HyperBRDF model...")
    model = HyperBRDF(in_features=6, out_features=3).to(device)
    print("Model created successfully")
    
    # Setup dataset
    print("Loading dataset...")
    dataset = MerlDataset(args.binary)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (model_input, gt) in enumerate(dataloader):
            # Move data to device
            model_input = {key: value.to(device) for key, value in model_input.items()}
            gt = {key: value.to(device) for key, value in gt.items()}
            
            # Forward pass
            optimizer.zero_grad()
            output = model(model_input)
            
            # Calculate loss
            img_loss = ((output['model_out'] - gt['amps']) ** 2).mean()
            latent_loss = args.kl_weight * torch.mean(output['latent_vec'] ** 2)
            
            # Total loss
            loss = img_loss + latent_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.6f}")
        
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch}/{args.epochs}], Average Loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pt')
    
    print("Training completed!")