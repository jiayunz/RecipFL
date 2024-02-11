"""
Graph HyperNetworks.

Some functionality in this script is based on:
https://github.com/facebookresearch/ppuda
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from .gnn import GatedGNN
from .layers import ShapeEncoder, TransformerEncoder, MLPDecoder, PosEnc, named_layered_modules
from .graph import Graph, GraphBatch, PRIMITIVES


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


class GHN(nn.Module):
    r"""
    Graph HyperNetwork based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    (https://arxiv.org/abs/1810.05749)
    """
    def __init__(self,
                 max_shape,
                 classifer_shape, # classifier head -> {key: shape}
                 max_param_len,
                 num_classes,
                 weight_norm=False,
                 ve=False,
                 layernorm=False,
                 node_hid=32,
                 debug_level=0):
        super(GHN, self).__init__()

        self.max_shape = max_shape
        self.classifer_shape = classifer_shape
        self.max_param_len = max_param_len
        self.layernorm = layernorm
        self.weight_norm = weight_norm
        self.ve = ve
        self.num_classes = num_classes
        self.node_hid = node_hid
        self.primitives_dict = {op: i for i, op in enumerate(PRIMITIVES)}
        self.debug_level = debug_level

        # client embedding
        self.client_enc = nn.Sequential(
            # nn.Embedding(num_embeddings=num_clients, embedding_dim=client_emb_hid),
            nn.Linear(num_classes, node_hid),
            nn.ReLU(inplace=True),
            nn.Linear(node_hid, node_hid * 2)
        )

        self.node_embed = torch.nn.Embedding(len(PRIMITIVES), node_hid)
        max_4dshape = [1, 1, 1, 1]
        for node_type, shape in self.max_shape.items():
            if len(shape) == 4:
                for i, s in enumerate(shape):
                    if s > max_4dshape[i]:
                        max_4dshape[i] = s

        self.shape_enc = ShapeEncoder(hid=node_hid,
                                      num_classes=num_classes,
                                      max_shape=(1, 1, *max_4dshape[2:]),
                                      debug_level=debug_level)
        self.weight_enc = TransformerEncoder(d_model=node_hid, max_len=max_param_len, nhead=4, num_encoder_layers=1, dim_feedforward=64, dropout=0.5)

        if layernorm:
            self.ln = nn.LayerNorm(node_hid)

        # encoder
        self.gnn = GatedGNN(in_features=node_hid, ve=ve)

        # decoder for each type of node
        for node_type in max_shape:
            decoder = MLPDecoder(
                in_features=node_hid,
                hid=(node_hid * 2, ),
                out_shape=max_shape[node_type],
                num_classes=num_classes
            )
            self.add_module(f'decoder_{node_type}', decoder)

        for k, s in classifer_shape.items():
            self.add_module(f"decoder_{k.replace('.', '@')}", nn.Sequential(nn.ReLU(), nn.Linear(node_hid * 2, np.prod(s))))

    @staticmethod
    def load(checkpoint_path, debug_level=1, device=default_device(), verbose=False):
        state_dict = torch.load(checkpoint_path, map_location=device)
        ghn = GHN(**state_dict['config'], debug_level=debug_level).to(device).eval()
        ghn.load_state_dict(state_dict['state_dict'])
        return ghn


    def forward(self, client_features, nets_torch, graphs=None, return_embeddings=False, predict_class_layers=True, bn_train=True):
        r"""
        Predict parameters for a list of >=1 networks.
        :param nets_torch: one network or a list of networks, each is based on nn.Module.
                           In case of evaluation, only one network can be passed.
        :param graphs: GraphBatch object in case of training.
                       For evaluation, graphs can be None and will be constructed on the fly given the nets_torch in this case.
        :param return_embeddings: True to return the node embeddings obtained after the last graph propagation step.
                                  return_embeddings=True is used for property prediction experiments.
        :param predict_class_layers: default=True predicts all parameters including the classification layers.
                                     predict_class_layers=False is used in fine-tuning experiments.
        :param bn_train: default=True sets BN layers in nets_torch into the training mode (required to evaluate predicted parameters)
                        bn_train=False is used in fine-tuning experiments
        :return: nets_torch with predicted parameters and node embeddings if return_embeddings=True
        """
        if not self.training:
            assert isinstance(nets_torch, nn.Module) or len(nets_torch) == 1, 'constructing the graph on the fly is only supported for a single network'

            if isinstance(nets_torch, list):
                nets_torch = nets_torch[0]

            if graphs is None:
                graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)])
                graphs.to_device(self.node_embed.weight.device)

        elif graphs is None:
            graphs = GraphBatch([Graph(net, ve_cutoff=50 if self.ve else 1) for net in nets_torch])
        graphs.to_device(self.node_embed.weight.device)

        # Find mapping between embeddings and network parameters
        param_groups, params_map, net_map, name_map, last_layer_indicator = self._map_net_params(graphs, nets_torch, self.debug_level > 0)

        # obtain initial embeddings for all nodes
        x = self.shape_enc(self.node_embed(graphs.node_feat[:, 0]), params_map, predict_class_layers=predict_class_layers)
        # obtain embeddings for pretrained weights
        weight_idx = {}
        for i in range(len(graphs.node_feat)):
            if i in name_map and name_map[i]['weight'] is not None:
                weight_idx[i] = len(weight_idx)
        weight_x = pad_sequence([name_map[i]['weight'] for i in range(len(graphs.node_feat)) if i in weight_idx], batch_first=True, padding_value=-1).transpose(1, 0).unsqueeze(-1)
        weight_emb = self.weight_enc(weight_x.to(self.node_embed.weight.device))
        weight_emb = torch.cat([weight_emb[weight_idx[i]].unsqueeze(0) if i in weight_idx else torch.zeros(1, self.node_hid).to(x.device) for i in range(len(graphs.node_feat))], axis=0)
        #x = torch.cat([x, weight_emb], dim=-1)
        x += weight_emb
        # obtain embedding for client
        if self.client_enc is not None:
            client_embs = self.client_enc(client_features.to(self.node_embed.weight.device))

        # Update node embeddings using a GatedGNN, MLP or another model
        x = self.gnn(x, graphs.edges, graphs.node_feat[:, 1])

        if self.layernorm:
            x = self.ln(x)

        # Predict max-sized parameters for a batch of nets using decoders
        for key, inds in param_groups.items():
            for ind in inds:
                matched, _, w_ind = params_map[ind]
                #print('ind', ind, matched['param_name'], {p: matched['module']._parameters[p].requires_grad for p in matched['module']._parameters})
                if (not name_map[ind]['requires_grad']) or w_ind is None:
                    continue  # e.g. pooling

                if name_map[ind]['param_name'] in self.classifer_shape:  # last weight
                    is_cls = True
                else:
                    is_cls = False

                for module_name, layer in self.named_modules():
                    if module_name == f"decoder_{name_map[ind]['name']}":
                        w = layer(x[[ind]], key, class_pred=is_cls)
                        if name_map[ind]['param_name'] in self.classifer_shape:
                            for class_pred_module_name, clss_pred_layer in self.named_modules():
                                if class_pred_module_name == f"decoder_{name_map[ind]['param_name'].replace('.', '@')}":
                                    if self.client_enc is not None:
                                        emb = client_embs[net_map[ind]].unsqueeze(0)
                                        #w = torch.cat([w, client_emb], dim=-1)
                                        w += emb
                                    w = clss_pred_layer(w).view(1, *self.classifer_shape[name_map[ind]['param_name']])
                                    break
                        break

                self._set_params(matched['module'], self._tile_params(w[0], matched['sz']), is_w=matched['is_w'])


        if not self.training and bn_train:

            def bn_set_train(module):
                if isinstance(module, nn.BatchNorm2d):
                    module.track_running_stats = False
                    module.training = True

            nets_torch.apply(bn_set_train)  # set BN layers to the training mode to enable evaluation without having running statistics

        if self.training:
            outputs = []
            for net in nets_torch:
                outputs.append({n: p for n, p in net.named_parameters() if p.requires_grad})
        else:
            outputs = {n: p for n, p in nets_torch.named_parameters() if p.requires_grad}

        return (nets_torch, x) if return_embeddings else outputs


    def _map_net_params(self, graphs, nets_torch, sanity_check=False):
        r"""
        Matches the parameters in the models with the nodes in the graph.
        Performs additional steps.
        :param graphs: GraphBatch object
        :param nets_torch: a single neural network of a list
        :param sanity_check:
        :return: mapping, params_map, net_map
        """
        mapping = {}
        params_map = {}
        net_map = {}
        name_map = {} # node_type in PRIMITIVES
        last_layer_indicator = {}

        nets_torch = [nets_torch] if type(nets_torch) not in [tuple, list] else nets_torch

        for b, (node_info, net) in enumerate(zip(graphs.node_info, nets_torch)):
            # target_modules = net.__dict__['_layered_modules'] if self.training else named_layered_modules(net)
            target_modules = named_layered_modules(net)
            requires_grad = {n: p.requires_grad for n, p in net.named_parameters()}
            params_weights = {k: v for (k, v) in net.state_dict().items()}

            # print(target_modules)
            param_ind = torch.sum(graphs.n_nodes[:b]).item()
            for cell_id in range(len(node_info)):
                max_node_ind = 0
                for (node_ind, p_, name, sz, last_weight, last_bias) in node_info[cell_id]:
                    if last_weight:
                        last_layer_indicator[param_ind + node_ind] = 1
                    elif last_bias:
                        last_layer_indicator[param_ind + node_ind] = -1
                    else:
                        last_layer_indicator[param_ind + node_ind] = 0

                    if node_ind > max_node_ind:
                        max_node_ind = node_ind
                    param_name = p_ if p_.endswith(('.weight', '.bias', 'in_proj_weight', 'in_proj_bias')) else p_ + '.weight'
                    p_weight = params_weights[param_name].view(-1) if param_name in params_weights else None
                    if p_weight is not None and p_weight.size(0) > self.max_param_len:
                        avg_p_weight = torch.zeros(self.max_param_len)
                        step = p_weight.size(0) // self.max_param_len
                        for i in range(self.max_param_len):
                            avg_p_weight[i] = torch.mean(p_weight[i * step : (i+1) * step])
                        p_weight = avg_p_weight
                    name_map[param_ind + node_ind] = {
                        'name': name,
                        'param_name': param_name,
                        'weight': p_weight,
                        'requires_grad': requires_grad[param_name] if param_name in requires_grad else False
                    }
                    try:
                        matched = [target_modules[cell_id][param_name]]
                    except:
                        matched = []

                    if len(matched) == 0:
                        if sz is not None:
                            params_map[param_ind + node_ind] = ({'sz': sz}, None, None)

                        if sanity_check:
                            for pattern in ['input', 'sum', 'concat', 'pool', 'glob_avg', 'msa', 'cse']:
                                good = name.find(pattern) >= 0
                                if good:
                                    break
                            assert good, \
                                (cell_id, param_name, name,
                                 node_info[cell_id],
                                 target_modules[cell_id])
                    else:
                        sz = matched[0]['sz']

                        def min_sz():
                            # to group predicted shapes and improve parallelization and at the same time not to predict much more than needed
                            min_size = np.ones_like(sz)
                            for i, s in enumerate(sz):
                                min_size[i] = min(sz[i], self.max_shape[name][i])
                            return tuple(min_size)

                        key = min_sz()

                        if key not in mapping:
                            mapping[key] = []
                        params_map[param_ind + node_ind] = (matched[0], key, len(mapping[key]))
                        mapping[key].append(param_ind + node_ind)
                        del target_modules[cell_id][param_name]

                # Prune redundant ops in Network by setting their params to None
                for m in target_modules[cell_id].values():
                    if m['is_w']:
                        m['module'].weight = None
                        if hasattr(m['module'], 'bias') and m['module'].bias is not None:
                            m['module'].bias = None

                for node_ind in range(max_node_ind+1):
                    net_map[param_ind + node_ind] = b

        return mapping, params_map, net_map, name_map, last_layer_indicator


    def _tile_params(self, w, target_shape):
        r"""
        Makes the shape of predicted parameter tensors the same as the target shape by tiling/slicing across channels dimensions.
        :param w: predicted tensor, for example of shape (64, 64, 11, 11)
        :param target_shape: tuple, for example (512, 256, 3, 3)
        :return: tensor of shape target_shape
        """
        t, s = target_shape, w.shape

        # Slice first to avoid tiling a larger tensor
        if len(t) == 1:
            if len(s) >= 1:
                w = w[:min(t[0], s[0])]
        elif len(t) == 2:
            if len(s) >= 2:
                w = w[:min(t[0], s[0]), :min(t[1], s[1])]
        elif len(t) == 3:
            if len(s) >= 3:
                w = w[:min(t[0], s[0]), :min(t[1], s[1]), :min(t[2], s[2])]
        else:
            w = w[:min(t[0], s[0]), :min(t[1], s[1]), :min(t[2], s[2]), :min(t[3], s[3])]

        s = w.shape
        assert len(s) == len(t), (s, t)

        # Tile out_channels
        if t[0] > s[0]:
            n_out = int(np.ceil(t[0] / s[0]))
            if len(t) == 1:
                w = w.repeat(n_out)[:t[0]]
            elif len(t) == 2:
                w = w.repeat((n_out, 1))[:t[0]]
            elif len(t) == 3:
                w = w.repeat((n_out, 1, 1))[:t[0]]
            else:
                w = w.repeat((n_out, 1, 1, 1))[:t[0]]

        # Tile in_channels
        if len(t) > 1:
            if t[1] > s[1]:
                n_in = int(np.ceil(t[1] / s[1]))
                if len(t) == 2:
                    w = w.repeat((1, n_in))[:, :t[1]]
                elif len(t) == 3:
                    w = w.repeat((1, n_in, 1))[:, t[1]]
                else:
                    w = w.repeat((1, n_in, 1, 1))[:, :t[1]]

        # Chop out any extra bits tiled
        if len(t) == 1:
            w = w[:t[0]]
        elif len(t) == 2:
            w = w[:t[0], :t[1]]
        elif len(t) == 3:
            w = w[:t[0], :t[1], :t[2]]
        else:
            w = w[:t[0], :t[1], :t[2], :t[3]]

        return w


    def _set_params(self, module, tensor, is_w):
        r"""
        Copies the predicted parameter tensor to the appropriate field of the module object.
        :param module: nn.Module
        :param tensor: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: the shape of the copied tensor
        """
        if self.weight_norm:
            tensor = self._normalize(module, tensor, is_w)
        is_layer_scale = hasattr(module, 'layer_scale') and module.layer_scale is not None
        key = ('layer_scale' if is_layer_scale else 'weight' ) if is_w else 'bias'
        target_param = getattr(module, key)
        sz_target = tuple(target_param) if isinstance(target_param, (list, tuple)) else target_param.shape
        if self.training:
            module.__dict__[key] = tensor  # set the value avoiding the internal logic of PyTorch
            # update parameters, so that named_parameters() will return tensors
            # with gradients (for multigpu and other cases)
            module._parameters[key] = tensor
        else:
            assert isinstance(target_param, nn.Parameter), type(target_param)
            # copy to make sure there is no sharing of memory
            target_param.data = tensor.clone()

        set_param = getattr(module, key)
        assert sz_target == set_param.shape, (sz_target, set_param.shape)
        return set_param.shape


    def _normalize(self, module, p, is_w):
        r"""
        Normalizes the predicted parameter tensor according to the Fan-In scheme described in the paper.
        :param module: nn.Module
        :param p: predicted tensor
        :param is_w: True if it is a weight, False if it is a bias
        :return: normalized predicted tensor
        """
        if p.dim() > 1:

            sz = p.shape

            if len(sz) > 2 and sz[2] >= 11 and sz[0] == 1:
                assert isinstance(module, PosEnc), (sz, module)
                return p    # do not normalize positional encoding weights

            no_relu = len(sz) > 2 and (sz[1] == 1 or sz[2] < sz[3])
            if no_relu:
                # layers not followed by relu
                beta = 1.
            else:
                # for layers followed by rely increase the weight scale
                beta = 2.

            # fan-out:
            # p = p * (beta / (sz[0] * p[0, 0].numel())) ** 0.5

            # fan-in:
            p = p * (beta / p[0].numel()) ** 0.5

        else:

            if is_w:
                p = 2 * torch.sigmoid(0.5 * p)  # BN/LN norm weight is [0,2]
            else:
                p = torch.tanh(0.2 * p)         # bias is [-1,1]

        return p