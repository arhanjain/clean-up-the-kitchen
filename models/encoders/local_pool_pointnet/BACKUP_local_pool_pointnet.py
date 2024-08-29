import torch
import torch.nn as nn
from robomimic.models.base_nets import ConvBase, Module
from . import ResidualBlockFC, UNet
from .unet3d import UNet3D
# There's no difference betweeen convbase and module functionally???

class LocalPoolPointnet(ConvBase):
    '''
    Based off the SCAR implementation of their visuomotor PCD policy.

    PointNet-based encoder network with ResNet blocks for each point.
    Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, 
                 c_dim=128, 
                 dim=3, 
                 hidden_dim=128, 
                 scatter_type='max', 
                 unet=False, 
                 unet_kwargs=None, 
                 unet3d=False, 
                 unet3d_kwargs=None, 
                 plane_resolution=None, 
                 grid_resolution=None, 
                 plane_type='xz', 
                 padding=0.1, 
                 n_blocks=5, 
                 local_coord=False, 
                 pos_encoding='linear', 
                 mc_vis=None, 
                 sparse=False
                 ):

        super().__init__()

        self.mc_vis = mc_vis
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            if sparse:
                from torchsparse.backbones import SparseResUNet42
                self.unet3d = SparseResUNet42(in_channels=32)
            else:
                self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        # if scatter_type == 'max':
        #     self.scatter = scatter_max
        # elif scatter_type == 'mean':
        #     self.scatter = scatter_mean
        # else:
        #     raise ValueError('incorrect scatter type')

        if local_coord:
            unit_size = 1.1 / self.reso_grid 
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None
        
        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2*hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2*hidden_dim)




    def output_shape(self, input_shape):
        #TODO 
        pass



class map2local(object):
    ''' Add new keys to the given input

    Args:
        s (float): the defined voxel size
        pos_encoding (str): method for the positional encoding, linear|sin_cos
    '''
    def __init__(self, s, pos_encoding='linear'):
        super().__init__()
        self.s = s
        self.pe = positional_encoding(basis_function=pos_encoding)

    def __call__(self, p):
        p = torch.remainder(p, self.s) / self.s # always possitive
        p = p - 0.5

        # print('using fmod instead of remainder in map2local!')
        # p = torch.fmod(p, self.s) / self.s # same sign as input p!

        p = self.pe(p)
        return p
