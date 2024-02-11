"""
Helper layers to build GHNs.

Some functionality in this script is based on:
https://github.com/facebookresearch/ppuda
"""

import numpy as np
import copy
import math
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from typing import Union


class ModuleLight(nn.Module):
    def __init__(self):
        super(ModuleLight, self).__init__()

    def __setattr__(self, name: str, value: Union[torch.Tensor, 'Module']) -> None:
        if isinstance(value, (list, torch.Tensor, nn.Parameter)) or (name in ['weight', 'bias'] and value is None):
            self._parameters[name] = tuple(value) if isinstance(value, list) else value
        else:
            object.__setattr__(self, name, value)

    def to(self, *args, **kwargs):
        return None

    def reset_parameters(self) -> None:
        return None


class LayerNormLight(ModuleLight):

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        super(LayerNormLight, self).__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]

        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        assert elementwise_affine
        self.weight = list(normalized_shape)
        self.bias = list(normalized_shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)


class BatchNorm2dLight(ModuleLight):

    def __init__(self, num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=False,
        device=None,
        dtype=None
    ):
        super(BatchNorm2dLight, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        assert affine and not track_running_stats, 'assumed affine and that running stats is not updated'
        self.running_mean = None
        self.running_var = None
        self.num_batches_tracked = None

        self.weight = [num_features]
        self.bias = [num_features]


    def forward(self, input: torch.Tensor) -> torch.Tensor:

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(
            input,
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


NormLayers = [nn.BatchNorm2d, nn.LayerNorm, BatchNorm2dLight, LayerNormLight]
try:
    import torchvision

    NormLayers.append(torchvision.models.convnext.LayerNorm2d)
except Exception as e:
    print(e, 'convnext requires torchvision >= 0.12, current version is ', torchvision.__version__)


class PosEnc(nn.Module):
    def __init__(self, C, ks, light=False):
        super().__init__()
        fn = torch.empty if light else torch.randn
        self.weight = nn.Parameter(fn(1, C, ks, ks))

    def forward(self, x):
        return x + self.weight


def named_layered_modules(model):
    if hasattr(model, 'module'):  # in case of multigpu model
        model = model.module
    layers = model._n_cells if hasattr(model, '_n_cells') else 1
    layered_modules = [{} for _ in range(layers)]
    cell_ind = 0
    for module_name, m in model.named_modules():
        cell_ind = m._cell_ind if hasattr(m, '_cell_ind') else cell_ind

        is_layer_scale = hasattr(m, 'layer_scale') and m.layer_scale is not None
        is_w = (hasattr(m, 'weight') and m.weight is not None) or is_layer_scale
        is_b = hasattr(m, 'bias') and m.bias is not None
        if is_w or is_b:
            if module_name.startswith('module.'):
                module_name = module_name[module_name.find('.') + 1:]
            if is_w:
                key = module_name + ('.layer_scale' if is_layer_scale else '.weight')
                w = m.layer_scale if is_layer_scale else m.weight
                #print('w.requires_grad', w.requires_grad)
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': True, #'requires_grad': m.weight.requires_grad,
                                                  'sz': tuple(w) if isinstance(w, (list, tuple)) else w.shape}
            if is_b:
                key = module_name + '.bias'
                b = m.bias
                #print('b.requires_grad', b.requires_grad)
                layered_modules[cell_ind][key] = {'param_name': key, 'module': m, 'is_w': False, #'requires_grad': m.bias.requires_grad,
                                                  'sz': tuple(b) if isinstance(b, (list, tuple)) else b.shape}

    return layered_modules


class ShapeEncoder(nn.Module):
    def __init__(self, hid, num_classes, max_shape, debug_level=0):
        super(ShapeEncoder, self).__init__()

        assert max_shape[2] == max_shape[3], max_shape
        self.debug_level = debug_level
        self.num_classes = num_classes
        self.ch_steps = (2**3, 2**6, 2**12, 2**13)
        self.channels = np.unique([1, 3, num_classes] +
                                  list(range(self.ch_steps[0], self.ch_steps[1], 2**3)) +
                                  list(range(self.ch_steps[1], self.ch_steps[2], 2**4)) +
                                  list(range(self.ch_steps[2], self.ch_steps[3] + 1, 2**5)))

        self.spatial = np.unique(list(range(1, max(12, max_shape[3]), 2)) + [14, 16])

        # create a look up dictionary for faster determining the channel shape index
        # include shapes not seen during training by assigning them the the closest seen values
        self.channels_lookup = {c: i for i, c in enumerate(self.channels)}
        self.channels_lookup_training = copy.deepcopy(self.channels_lookup)
        for c in range(4, self.ch_steps[0]):
            self.channels_lookup[c] = self.channels_lookup[self.ch_steps[0]]  # 4-7 channels will be treated as 8 channels
        for c in range(1, self.channels[-1]):
            if c not in self.channels_lookup:
                self.channels_lookup[c] = self.channels_lookup[self.channels[np.argmin(abs(self.channels - c))]]

        self.spatial_lookup = {c: i for i, c in enumerate(self.spatial)}
        self.spatial_lookup_training = copy.deepcopy(self.spatial_lookup)
        self.spatial_lookup[2] = self.spatial_lookup[3]  # 2x2 (not seen during training) will be treated as 3x3
        for c in range(1, self.spatial[-1]):
            if c not in self.spatial_lookup:
                self.spatial_lookup[c] = self.spatial_lookup[self.spatial[np.argmin(abs(self.spatial - c))]]

        n_ch, n_s = len(self.channels), len(self.spatial)
        self.embed_spatial = torch.nn.Embedding(n_s + 1, hid // 4)
        self.embed_channel = torch.nn.Embedding(n_ch + 1, hid // 4)

        self.register_buffer('dummy_ind', torch.tensor([n_ch, n_ch, n_s, n_s], dtype=torch.long).view(1, 4),
                             persistent=False)


    def forward(self, x, params_map, predict_class_layers=True):
        shape_ind = self.dummy_ind.repeat(len(x), 1)

        self.printed_warning = False
        for node_ind in params_map:
            sz = params_map[node_ind][0]['sz']
            if sz is None:
                continue

            sz_org = sz
            if len(sz) == 1:
                sz = (sz[0], 1)
            if len(sz) == 2:
                sz = (sz[0], sz[1], 1, 1)
            if len(sz) == 3:
                sz = (sz[0], sz[1], sz[2], 1)
            assert len(sz) == 4, sz

            if not predict_class_layers and params_map[node_ind][1] in ['cls_w', 'cls_b']:
                # keep the classification shape as though the GHN is used on the dataset it was trained on
                sz = (self.num_classes, *sz[1:])

            recognized_sz = 0
            for i in range(4):
                # if not in the dictionary, then use the maximum shape
                if i < 2:  # for out/in channel dimensions
                    shape_ind[node_ind, i] = self.channels_lookup[sz[i] if sz[i] in self.channels_lookup else self.channels[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.channels_lookup_training)
                else:  # for kernel height/width
                    shape_ind[node_ind, i] = self.spatial_lookup[sz[i] if sz[i] in self.spatial_lookup else self.spatial[-1]]
                    if self.debug_level and not self.printed_warning:
                        recognized_sz += int(sz[i] in self.spatial_lookup_training)

            if self.debug_level and not self.printed_warning:  # print a warning once per architecture
                if recognized_sz != 4:
                    print( 'WARNING: unrecognized shape %s, so the closest shape at index %s will be used instead.' % (
                        sz_org, ([self.channels[c.item()] if i < 2 else self.spatial[c.item()] for i, c in
                                  enumerate(shape_ind[node_ind])])))
                    self.printed_warning = True

        shape_embed = torch.cat(
            (self.embed_channel(shape_ind[:, 0]),
             self.embed_channel(shape_ind[:, 1]),
             self.embed_spatial(shape_ind[:, 2]),
             self.embed_spatial(shape_ind[:, 3])), dim=1)

        return x + shape_embed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, max_len, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.nhead = nhead
        if d_model % self.nhead != 0:
            d_model = d_model + self.nhead - d_model % self.nhead

        self.d_model = d_model
        self.enc_embedding = nn.Linear(1, d_model)
        self.pos_embedding_enc = PositionalEncoding(d_model, dropout, max_len)

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, norm=encoder_norm)

    def make_src_mask(self, inp):
        return torch.all(inp == -1, dim=-1).transpose(0, 1)

    def forward(self, src, output_type='avg'):
        # src: [src_len, batch_size, feature_dim]
        src_pad_mask = self.make_src_mask(src)

        src = self.enc_embedding(src)
        src = self.pos_embedding_enc(src)  # [src_len, batch_size, embed_dim]
        memory = self.encoder(src=src, mask=None, src_key_padding_mask=src_pad_mask) # padding marker
        seq_len = (~src_pad_mask).sum(-1)
        memory = torch.mul(memory, ~src_pad_mask.repeat(self.d_model, 1, 1).permute(2, 1, 0))

        # [src_len, batch_size, embed_dim]
        if output_type == 'sum':
            embedding = torch.sum(memory, dim=0)
        elif output_type == 'avg':
            embedding = torch.sum(memory, dim=0) / seq_len.unsqueeze(-1)
        elif output_type == 'last':
            embedding = memory[[(seq_len-1).to(torch.long), torch.range(0, memory.size(1)-1).to(torch.long)]]  # the last timestep
        else:
            raise ValueError('Wrong value of output_type.')


        return embedding  # [batch_size, emb_dim]


def get_activation(activation):
    if activation is not None:
        if activation == 'relu':
            f = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            f = nn.LeakyReLU()
        elif activation == 'selu':
            f = nn.SELU()
        elif activation == 'elu':
            f = nn.ELU()
        elif activation == 'rrelu':
            f = nn.RReLU()
        elif activation == 'sigmoid':
            f = nn.Sigmoid()
        else:
            raise NotImplementedError(activation)
    else:
        f = nn.Identity()

    return f


class MLP(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(32, 32),
                 activation='relu',
                 last_activation='same'):
        super(MLP, self).__init__()

        assert len(hid) > 0, hid
        fc = []
        for j, n in enumerate(hid):
            fc.extend([nn.Linear(in_features if j == 0 else hid[j - 1], n),
                       get_activation(last_activation if
                                      (j == len(hid) - 1 and
                                       last_activation != 'same')
                                      else activation)])
        self.fc = nn.Sequential(*fc)


    def forward(self, x, *args, **kwargs):
        if isinstance(x, tuple):
            x = x[0]
        return self.fc(x)


class MLPDecoder(nn.Module):
    def __init__(self,
                 in_features=32,
                 hid=(64,),
                 out_shape=None,
                 num_classes=None):
        super(MLPDecoder, self).__init__()

        assert len(hid) > 0, hid
        self.out_shape = out_shape
        self.num_classes = num_classes
        self.mlp = MLP(in_features=in_features,
                       hid=(*hid, np.prod(out_shape)),
                       activation='relu',
                       last_activation=None)


    def forward(self, x, max_shape=(1,1,1,1), class_pred=False):
        if class_pred:
            x = list(self.mlp.fc.children())[0](x)  # shared first layer
        else:
            x = self.mlp(x).view(-1, *self.out_shape)
            if sum(max_shape[2:]) > 0:
                x = x[:, :, :, :max_shape[2], :max_shape[3]]
        return x