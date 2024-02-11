import os
import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import numpy as np

from utils.general_utils import set_seed
from server import Server
from utils.logger import Logger

def main(args):
    save_dir = os.path.join(args.save_dir, f"{args.task}/seed{args.seed}/")
    args.data_shares = [(1 - args.large_share) / (args.total_clients - args.n_large)] * (args.total_clients - args.n_large) + [args.large_share / args.n_large] * args.n_large
    assert round(np.sum(args.data_shares), 2) == 1., args.data_shares
    assert args.total_clients == len(args.data_shares)
    assert args.finetune_epochs <= args.epochs

    args.sample_clients = min(args.total_clients, args.sample_clients)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_path = os.path.join(save_dir, f"alpha{args.alpha}.ghn{'.distill' * args.distill}{'.scaled'*(args.scaling=='width')}.tc{args.total_clients}.sc{args.sample_clients}.d{str(args.large_share)}.log")  # + datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))
    args.logger = Logger(file_path=log_path).get_logger()
    args.logger.critical(log_path)
    torch.cuda.set_device(args.gpu)
    if args.device is None:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    set_seed(args.seed)
    if args.task == 'cifar10':
        small_model_name = 'ResNet_1block' if args.scaling == 'depth' else 'ResNet18_scaled'
        args.client_model_names = {i: [small_model_name] for i in range(args.total_clients-1)}
        for i in range(args.n_large):
            args.client_model_names[args.total_clients - args.n_large + i] = ['ResNet18', small_model_name] #
    elif args.task == 'cifar100':
        small_model_name = 'DenseNet_1block' if args.scaling == 'depth' else 'DenseNet121_scaled'
        args.client_model_names = {i: [small_model_name] for i in range(args.total_clients-1)}
        for i in range(args.n_large):
            args.client_model_names[args.total_clients - args.n_large + i] = ['DenseNet121', small_model_name]
    elif args.task == 'mnist':
        args.client_model_names = {i: ['LeNet5_dwscaled'] for i in range(args.total_clients-1)}
        for i in range(args.n_large):
            args.client_model_names[args.total_clients - args.n_large + i] = ['LeNet5', 'LeNet5_dwscaled']
    elif args.task == 'mnli':
        args.client_model_names = {i: ['DistilBERT'] for i in range(args.total_clients-1)}
        for i in range(args.n_large):
            args.client_model_names[args.total_clients - args.n_large + i] = ['BERT', 'DistilBERT']

    args.metrics = ['ACC']

    if args.task == 'cifar10':
        args.n_class = 10
    elif args.task == 'cifar100':
        args.n_class = 100
    elif args.task == 'mnist':
        args.n_class = 10
    elif args.task == 'mnli':
        args.n_class = 3
        args.epochs = 1

    args.logger.critical(args)

    server = Server(args)
    args.logger.debug('Server created.')

    for client_id, (client_ip, client_port) in client_addr.items():
        server.register_client(client_id, client_ip, client_port)

    server.train(args)

    del server

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=4321, help="random seed")
    parser.add_argument('-g', '--gpu', type=int, default="7", help="gpu id")
    # training & communication
    parser.add_argument('-p', '--port', type=int, default=12345, help="server port")
    parser.add_argument('--client_ip', type=str,  help="client IP, see output of run_client.py")
    parser.add_argument('--cp', action='append', help="client ports")
    parser.add_argument('--save_dir', type=str, default="logs/")
    parser.add_argument('--device', choices=['cuda', 'cpu'], help="use cuda or cpu")
    parser.add_argument('--buffer_size', type=int, default=1048576)
    parser.add_argument('--timeout', type=int, default=7200)
    # configuration
    parser.add_argument('-t', '--task', choices=['cifar10', 'cifar100', 'mnist', 'mnli'], default='cifar10', help="task name")
    parser.add_argument('--scaling', choices=['width', 'depth', 'architecture'], default='width', help="model scaling strategy")
    parser.add_argument('--n_large', type=int, help="number of large devices")
    parser.add_argument('--large_share', type=float, default=0.5, help="percentage of data on the large device")
    parser.add_argument('--alpha', type=float, default=0.5, help="alpha for dirichlet distribution")
    parser.add_argument('--total_clients', type=int, default=None, help="number of total clients")
    parser.add_argument('--sample_clients', type=int, default=10, help="number of clients join training at each round")
    parser.add_argument('-e', '--epochs', type=int, default=5, help="number of training epochs per round")
    parser.add_argument('--finetune_epochs', type=int, default=1, help="number of training epochs per round")
    parser.add_argument('-r', '--rounds', type=int, default=50, help="number of communication rounds")
    # model parameter
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate of hypernet")
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--node_hid', type=int, default=128, help="node embedding dimension")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.task == 'cifar10':
        args.rounds = 50
    elif args.task == 'cifar100':
        args.rounds = 500
    elif args.task == 'mnist':
        args.rounds = 100
        args.scaling = 'architecture'
    elif args.task == 'mnli':
        args.rounds = 50
        args.scaling = 'architecture'

    if args.total_clients is None: # default
        if args.task == 'cifar10':
            args.total_clients = 5+1
        elif args.task == 'cifar100':
            args.total_clients = 50 + 1
        elif args.task == 'mnist':
            args.total_clients = 500 + 2
        elif args.task == 'mnli':
            args.total_clients = 21

    if args.n_large is None: # default
        if args.task == 'cifar10':
            args.n_large = 1
        elif args.task == 'cifar100':
            args.n_large = 1
        elif args.task == 'mnist':
            args.n_large = 2
        elif args.task == 'mnli':
            args.n_large = 1

    client_clusters = [(args.client_ip, int(p)) for p in args.cp]

    client_addr = {i: client_clusters[i % len(client_clusters)] for i in range(args.total_clients - args.n_large)}
    for i in range(args.n_large):
        client_addr[args.total_clients-args.n_large+i] = client_clusters[-((i+1) % len(client_clusters))]

    main(args)
