import warnings
import random
import numpy as np
from torch.utils.data import random_split, Subset
from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision import transforms
from copy import deepcopy

IMAGE_SIZE = 32


def load_cifar(data_name, data_path, data_shares, alpha, n_large):
    # alpha: parameter for dirichlet distribution
    small_clients = [c for c in range(len(data_shares) - n_large)]
    subsets = []
    datasets = get_datasets(data_name, data_path)

    num_classes, num_samples, data_labels_list = get_num_classes_samples(datasets[0])
    # only generate q_class for small devices
    q_class = np.random.dirichlet([alpha] * num_classes, len(small_clients))

    for i, d in enumerate(datasets):
        num_classes, num_samples, data_labels_list = get_num_classes_samples(d)
        # for large device, make the data distribution as the universal distribution
        data_class_idx = {i: np.where(data_labels_list == i)[0] for i in range(num_classes)}
        for data_idx in data_class_idx.values():
            random.shuffle(data_idx)

        total_cnt = np.sum(num_samples)
        usr_subset_idx = [[] for i in data_shares]
        large_data_cnt = 0
        for usr_i, share in enumerate(data_shares):
            if usr_i not in small_clients:
                for c in range(num_classes):
                    if i == 0:
                        end_idx = int(num_samples[c] * data_shares[usr_i])
                    else:
                        end_idx = int(num_samples[c] / len(data_shares))  # client test data has the same amount
                    usr_subset_idx[usr_i].extend(data_class_idx[c][:end_idx])
                    data_class_idx[c] = data_class_idx[c][end_idx:]
                    num_samples[c] -= end_idx
                    large_data_cnt += end_idx

        if i == 0: # trainset
            total_small_ratio = sum(data_shares[:-n_large])
            if total_small_ratio > 1 - sum(data_shares[-n_large:]):
                warnings.warn(f'total_small_ratio {total_small_ratio} > 1 - total_large_ratio {1 - sum(data_shares[-n_large:])}.')
            elif total_small_ratio < 1 - sum(data_shares[-n_large:]):
                print('total_small_ratio < 1 - total_large_ratio. Adjust dataset')
                for c in range(num_classes):
                    num_samples[c] = int(num_samples[c] * total_small_ratio / (1 - sum(data_shares[-n_large:])))
                    data_class_idx[c] = data_class_idx[c][:num_samples[c]]
                print('small ratio', np.sum(num_samples)/total_cnt, 'large ratio:', large_data_cnt/total_cnt)

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

    trainData, valData, testData = subsets[0], subsets[1], subsets[2]

    return trainData, valData, testData


def get_datasets(data_name, dataroot, val_size=10000):
    """
    get_datasets returns train/val/test data splits of CIFAR10/100 datasets
    :param data_name: name of dataset, choose from [cifar10, cifar100]
    :param dataroot: root to data dir
    :param normalize: True/False to normalize the data
    :param val_size: validation split size (in #samples)
    :return: train_set, val_set, test_set (tuple of pytorch dataset/subset)
    """

    if data_name =='cifar10':
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
    elif data_name == 'cifar100':
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100
    else:
        raise ValueError("choose data_name from ['mnist', 'cifar10', 'cifar100']")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalization
    ])

    dataset = data_obj(
        dataroot,
        train=True,
        download=True,
        transform=transform_train
    )

    test_set = data_obj(
        dataroot,
        train=False,
        download=True,
        transform=transform_test
    )

    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set, test_set


def get_num_classes_samples(dataset):
    """
    extracts info about certain dataset
    :param dataset: pytorch dataset object
    :return: dataset info number of classes, number of samples, list of labels
    """
    # ---------------#
    # Extract labels #
    # ---------------#
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


def gen_data_split(data_class_idx, q_class, q_client, k_samples_at_a_time=5):
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

    while np.sum(num_samples_clients) != 0:
        # iterate clients
        curr_cid = np.random.randint(len(q_client))
        # If current node is full resample a client
        if num_samples_clients[curr_cid] <= 0:
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
