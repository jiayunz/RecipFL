import os
import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from copy import deepcopy
from transformers import BertTokenizerFast
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MAX_LENGTH = 100
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

class MNLIDataset(Dataset):
    def __init__(self, premise, hypothesis, targets):
        self.targets = targets

        premise = tokenizer.batch_encode_plus(premise, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH, return_token_type_ids=False, return_attention_mask=False)['input_ids']
        hypothesis = tokenizer.batch_encode_plus(hypothesis,add_special_tokens=False, truncation=True, max_length=MAX_LENGTH, return_token_type_ids=False, return_attention_mask=False)['input_ids']
        self.token_ids = []
        self.mask_ids = []
        self.seg_ids = []

        for p_id, h_id in zip(premise, hypothesis):
            pair_token_ids = [tokenizer.cls_token_id] + p_id + [tokenizer.sep_token_id] + h_id + [tokenizer.sep_token_id]
            premise_len = len(p_id)
            hypothesis_len = len(h_id)

            segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

            self.token_ids.append(torch.tensor(pair_token_ids))
            self.seg_ids.append(segment_ids)
            self.mask_ids.append(attention_mask_ids)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.seg_ids[idx], self.mask_ids[idx], self.targets[idx]


def collate_fn(batch):
    batch_token_ids, batch_seg_ids, batch_mask_ids, batch_labels = zip(*batch)

    batch_token_ids = pad_sequence([torch.LongTensor(x) for x in batch_token_ids], batch_first=True)
    batch_seg_ids = pad_sequence([torch.LongTensor(x) for x in batch_seg_ids], batch_first=True)
    batch_mask_ids = pad_sequence([torch.LongTensor(x) for x in batch_mask_ids], batch_first=True)

    batch_labels = torch.FloatTensor(batch_labels)

    return (batch_token_ids, batch_seg_ids, batch_mask_ids), batch_labels


def get_datasets(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'multinli_1.0_train.txt'), sep='\t', on_bad_lines='skip')[['gold_label', 'genre', 'sentence1', 'sentence2']]
    test_df = pd.read_csv(os.path.join(data_path, 'multinli_1.0_dev_matched.txt'), sep='\t', on_bad_lines='skip')[['gold_label', 'genre', 'sentence1', 'sentence2']]
    train_df = train_df[train_df['gold_label'] != '-']
    test_df = test_df[test_df['gold_label'] != '-']
    label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    train_df['gold_label'] = train_df['gold_label'].apply(lambda x: label_dict[x])
    test_df['gold_label'] = test_df['gold_label'].apply(lambda x: label_dict[x])

    trainset = MNLIDataset(train_df['sentence1'].astype(str).tolist(), train_df['sentence2'].astype(str).tolist(), train_df['gold_label'].tolist())
    testset = MNLIDataset(test_df['sentence1'].astype(str).tolist(), test_df['sentence2'].astype(str).tolist(), test_df['gold_label'].tolist())

    return trainset, testset


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    if isinstance(dataset, Subset):
        if isinstance(dataset.dataset.targets, list):
            data_labels_list = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            data_labels_list = dataset.dataset.targets[dataset.indices]
    else:
        if isinstance(dataset.targets, list):
            data_labels_list = np.array(dataset.targets)
        else:
            data_labels_list = dataset.targets
    classes, num_samples = np.unique(data_labels_list, return_counts=True)
    num_classes = len(classes)
    return num_classes, num_samples, data_labels_list


def load_mnli(data_path, data_shares, alpha, n_large):
    # alpha: parameter for dirichlet distribution
    small_clients = [c for c in range(len(data_shares) - n_large)]
    subsets = []
    datasets = get_datasets(data_path)

    num_classes, num_samples, data_labels_list = get_num_classes_samples(datasets[0])
    # only generate q_class for small devices
    q_class = np.random.dirichlet([alpha] * num_classes, len(small_clients))

    for i, d in enumerate(datasets):
        num_classes, num_samples, data_labels_list = get_num_classes_samples(d)
        # for large device, make the data distribution as the universal distribution
        data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}
        for data_idx in data_class_idx.values():
            random.shuffle(data_idx)

        usr_subset_idx = [[] for i in data_shares]
        for usr_i, share in enumerate(data_shares):
            if usr_i not in small_clients:
                for c in range(num_classes):
                    if i == 0:
                        end_idx = int(num_samples[c] * data_shares[usr_i] / np.sum(data_shares))
                    else:
                        end_idx = int(num_samples[c] / len(data_shares))  # client test data has the same amount
                    usr_subset_idx[usr_i].extend(data_class_idx[c][:end_idx])
                    data_class_idx[c] = data_class_idx[c][end_idx:]
                    num_samples[c] -= end_idx

        if i == 0:
            q_client = np.array(data_shares, dtype=float)[small_clients] / np.sum([data_shares[c] for c in small_clients])
        else:
            q_client = np.ones_like(small_clients) / np.sum([data_shares[c] for c in small_clients])
        small_usr_subset_idx = gen_data_split(data_class_idx, q_class, q_client)

        # create subsets for each client
        small_id = 0
        for user_i in range(len(data_shares)):
            if not len(usr_subset_idx[user_i]):
                usr_subset_idx[user_i] = small_usr_subset_idx[small_id]
                small_id += 1

        subsets.append(list(map(lambda x: Subset(d, x), usr_subset_idx)))

    trainData, testData = subsets[0], subsets[1]

    return trainData, testData


def gen_data_split(data_class_idx, q_class, q_client, k_samples_at_a_time=10):
    """Non-iid Dirichlet partition.
    The method is from The method is from paper `Federated Learning Based on Dynamic Regularization <https://openreview.net/forum?id=B7v4QMR6Z9w>`_.
    This function can be used by given specific sample number for all clients.
    Args:
        :param q_class: class distribution at each client
        :param q_client: sample distribution cross clients
    Returns:
        dict: ``{ client_id: indices}``.
    """
    num_samples = np.array([len(indices) for cls, indices in data_class_idx.items()])
    num_samples_clients = (q_client * num_samples.sum()).round().astype(int)
    delta_data = num_samples.sum() - num_samples_clients.sum()
    client_id = 0
    for i in range(abs(delta_data)):
        num_samples_clients[client_id % len(q_client)] += np.sign(delta_data)
        client_id += 1

    # Create class index mapping
    data_class_idx = {cls: set(data_class_idx[cls]) for cls in data_class_idx}

    q_class_cumsum = np.cumsum(q_class, axis=1) # cumulative sum
    num_samples_tilde = deepcopy(num_samples)

    client_indices = [[] for _ in range(len(q_client))]

    candidate_clients = set(range(len(q_client)))

    while np.sum(num_samples_clients) != 0:
        # iterate clients
        curr_cid = np.random.choice(list(candidate_clients))
        # If current node is full resample a client
        if num_samples_clients[curr_cid] <= 0:
            candidate_clients.remove(curr_cid)
            continue

        while True:
            curr_class = np.argmax((np.random.uniform() <= q_class_cumsum[curr_cid]) & (num_samples_tilde > 0))
            # Redraw class label if no rest in current class samples
            if num_samples_tilde[curr_class] <= 0:
                continue

            k_samples = min(k_samples_at_a_time, len(data_class_idx[curr_class]))
            random_sample_idx = np.random.choice(list(data_class_idx[curr_class]), k_samples, replace=False)
            num_samples_tilde[curr_class] -= k_samples
            num_samples_clients[curr_cid] -= k_samples

            client_indices[curr_cid].extend(list(random_sample_idx))
            data_class_idx[curr_class] -= set(random_sample_idx)
            break

    client_dict = [client_indices[cid] for cid in range(len(q_client))]
    return client_dict
