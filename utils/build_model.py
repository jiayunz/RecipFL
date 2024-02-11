import numpy as np
import torch
from transformers import AutoAdapterModel, BertTokenizerFast

from .models.lenet import LeNet5, LeNet5_dwscaled
from .models.resnet import ResNet18, ResNet_1block, ResNet18_wscaled, ResNet101
from .models.densenet import DenseNet121, DenseNet121_1block, DenseNet121_wscaled
from .models.vgg import VGG

def build_model(model_name, task, n_class, device):
    if task.startswith('cifar'):
        dummy_input = torch.randn(1, 3, 32, 32, device=device)
    elif task == 'mnist':
        dummy_input = torch.randn(1, 1, 28, 28, device=device)
    elif task == 'mnli':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        p_id = np.random.randint(100, 10000, 10).tolist()
        h_id = np.random.randint(100, 10000, 10).tolist()
        pair_token_ids = torch.LongTensor([[tokenizer.cls_token_id] + p_id + [
            tokenizer.sep_token_id] + h_id + [tokenizer.sep_token_id]]).to(device)
        segment_ids = torch.LongTensor([[0] * (len(p_id) + 2) + [1] * (len(h_id) + 1)]).to(device)
        attention_mask_ids = torch.LongTensor([[1] * (len(p_id) + len(h_id) + 3)]).to(device)
        dummy_input = (pair_token_ids, segment_ids, attention_mask_ids)

    if model_name == 'ResNet18':
        model = ResNet18(dummy_input.size(1), n_class)
    elif model_name == 'ResNet_1block':
        model = ResNet_1block(num_blocks=[2], input_channel=dummy_input.size(1), num_classes=n_class)
    elif model_name == 'ResNet18_scaled':
        model = ResNet18_wscaled(dummy_input.size(1), n_class, scale_ratio=0.2)
    elif model_name == 'ResNet101':
        model = ResNet101(dummy_input.size(1), n_class)
    elif model_name == 'VGG16':
        model = VGG('VGG11', n_class, dummy_input.size(1))
    elif model_name == 'DenseNet121':
        model = DenseNet121(n_class)
    elif model_name == 'DenseNet121_1block':
        model = DenseNet121_1block(n_class)
    elif model_name == 'DenseNet121_scaled':
        model = DenseNet121_wscaled(n_class, scale_ratio=0.5)
    elif model_name == 'LeNet5':
        model = LeNet5(n_class, dummy_input.size(1))
    elif model_name == 'LeNet5_dwscaled':
        model = LeNet5_dwscaled(n_class)
    elif model_name == 'BERT':
        model = AutoAdapterModel.from_pretrained('bert-base-uncased')
        model.add_adapter("mnli")
        model.add_classification_head("mnli", num_labels=n_class)
        model.active_adapters = "mnli"
    elif model_name == 'DistilBERT':
        model = AutoAdapterModel.from_pretrained("distilbert-base-uncased")
        model.add_adapter("mnli")
        model.add_classification_head("mnli", num_labels=n_class)
        model.active_adapters = "mnli"
    else:
        raise ValueError('Wrong model name:', model_name)


    model.dummy_input = dummy_input

    return model

