import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WeightAveraging():
    def __init__(self, param_keys, client_model_names):
        self.param_keys = param_keys
        self.client_model_names = client_model_names

    def update(self, orig_model_weights, client_model_weights, client_weights):
        # orig_model_weights: global model weights
        # client_weights: number of data samples
        updated_model_weights = {}
        for k, p in orig_model_weights.items():
            new_p = self.update_k(k, p, client_model_weights, client_weights)
            if new_p is None:
                new_p = p
            updated_model_weights[k] = new_p

        return updated_model_weights

    def update_k(self, k, orig_p, new_model_weights, client_weights):
        k_update_weights = []
        stacked_model_weights = []
        for c, models in new_model_weights.items():
            for model_name, model_weights in zip(self.client_model_names[c], models):
                if k in model_weights and orig_p.size() == model_weights[k].size():
                    k_update_weights.append(client_weights[c])
                    stacked_model_weights.append(model_weights[k].cpu())
                elif k.startswith(model_name):
                    model_k = k[len(model_name) + 1:]
                    if model_k in model_weights and orig_p.size() == model_weights[model_k].size():
                        k_update_weights.append(client_weights[c])
                        stacked_model_weights.append(model_weights[model_k].cpu())

        if not len(k_update_weights):
            return

        k_update_weights = torch.FloatTensor(k_update_weights)
        k_update_weights /= k_update_weights.sum()
        for _ in orig_p.shape:
            k_update_weights = k_update_weights.unsqueeze(-1)

        return (k_update_weights * torch.stack(stacked_model_weights, 0)).sum(0).to(orig_p.device)


def convert_model_key_to_idx(global_key_to_idx, model_name, model_weights):
    converted_model_weights = {}
    for k, p in model_weights.items():
        if k in global_key_to_idx:
            global_k_idx = global_key_to_idx[k]
        elif model_name + '.' + k in global_key_to_idx:
            global_k_idx = global_key_to_idx[model_name + '.' + k]
        else:
            continue

        converted_model_weights[global_k_idx] = p.cpu()

    return converted_model_weights


def prepare_client_weights(model, model_name, recv_weight):
    new_model_weights = {}
    for k, p in model.state_dict().items():
        if model_name + '.' + k in recv_weight:
            global_k = model_name + '.' + k
            if p.size() == recv_weight[global_k].size():
                new_model_weights[k] = recv_weight[global_k].cpu()
        elif k in recv_weight and p.size() == recv_weight[k].size():
            new_model_weights[k] = recv_weight[k].cpu()

    return new_model_weights
