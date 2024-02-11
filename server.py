import torch
from torch import nn
import numpy as np
import pickle
from copy import deepcopy
from collections import defaultdict, OrderedDict
import time
import socket
import threading
from communication_utils import recv, send

from utils.build_model import build_model
from utils.general_utils import prepare_client_weights, convert_model_key_to_idx, WeightAveraging
from utils.hypernetwork.graph import Graph, GraphBatch
from utils.hypernetwork.nn import GHN

EPS = 1e-7

class Server():
    def __init__(self, args):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.port = args.port
        self.total_clients = args.total_clients
        self.buffer_size = args.buffer_size
        self.timeout = args.timeout
        self.device = args.device
        self.metrics = args.metrics
        self.client_clusters = defaultdict(set)
        self.client_addr = {}
        self.client_features = {}
        self.logger = args.logger
        self.client_model_names = args.client_model_names

        # the first in the list is the final evaluated one
        self.client_models = defaultdict(list)
        for cid, model_names in args.client_model_names.items():
            for model_name in model_names:
                self.client_models[cid].append(build_model(model_name, args.task, args.n_class, args.device))

        if args.task == 'mnli':
           args.classifier_name = 'heads.'
        else:
           args.classifier_name = 'classifier.'

        self.graphs = defaultdict(list)
        node_type_map = {}
        max_shape = {}  # node_type -> max_shape
        classifer_shape = {}
        max_param_len = 0
        for c, models in self.client_models.items():
            for model in models:
                graph = Graph(model.to(self.device), ve_cutoff=50)
                self.graphs[c].append(graph)

                for cell_id in range(len(graph.node_info)):
                    for (node_ind, p_, name, sz, last_weight, last_bias) in graph.node_info[cell_id]:
                        param_name = p_ if p_.endswith(('.weight', '.bias', 'in_proj_weight', 'in_proj_bias')) else p_ + '.weight'
                        node_type_map[param_name] = name

                if args.task == 'mnli':
                    # only train adapter and classification head
                    model.train_adapter("mnli")

                for k, p in model.named_parameters():
                    # separately set for BN/LN biases as they are
                    # not represented as separate nodes in graphs
                    if len(p.size()) == 1 and k not in node_type_map:
                        assert k.endswith('.bias') >= 0, k
                        node_type_map[k] = 'bias'

                    if k.startswith(args.classifier_name):
                        classifer_shape[k] = p.size()

                    max_param_len = max(np.prod(p.size()), max_param_len)
                    if k in node_type_map and node_type_map[k] not in max_shape:
                        max_shape[node_type_map[k]] = np.ones_like(p.size())

                    if not p.requires_grad:
                        continue

                    for i, s in enumerate(p.size()):
                        if s > max_shape[node_type_map[k]][i]:
                            max_shape[node_type_map[k]][i] = s

        self.logger.debug(f'max_shape: {max_shape}, max_param_len: {max_param_len}')

        self.hypernet = GHN(
            max_shape={node: tuple([min(s, 10240) for s in shape]) for node, shape in max_shape.items()},
            classifer_shape=classifer_shape,
            max_param_len=128,
            num_classes=args.n_class,
            weight_norm=True,
            ve=True,
            layernorm=True,
            node_hid=args.node_hid).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.hypernet.parameters(), lr=args.lr)

        # create global_model_weights after activate adapters
        self.global_model_weights = OrderedDict()
        for cid, model_names in args.client_model_names.items():
            for i, model_name in enumerate(model_names):
                for k, p in self.client_models[cid][i].named_parameters():
                    if not p.requires_grad:
                        continue
                    if k not in self.global_model_weights:
                        self.global_model_weights[k] = p.cpu()
                    elif p.size() != self.global_model_weights[k].size():
                        global_k = model_name + '.' + k
                        self.global_model_weights[global_k] = p.cpu()

        self.global_keys = list(self.global_model_weights.keys())
        self.global_key_to_idx = {global_k: i for i, global_k in enumerate(self.global_keys)}

        client_model_sizes = defaultdict(list)
        for c in range(self.total_clients):
            if c < self.total_clients - args.n_large:
                client_model_sizes[c] = ['small']
            else:
                client_model_sizes[c] = ['large', 'small']
        self.aggregator = WeightAveraging(self.global_keys, self.client_model_names)

    def register_client(self, id, ip, port):
        self.client_addr[id] = (ip, port)
        self.client_clusters[(ip, port)].add(id)

    def server_update(self, init_client_weights, client_model_weights, client_update_weights, unweighted=True):
        self.hypernet.train()
        self.optimizer.zero_grad()

        if unweighted:
            hypernet_outputs = []
            delta_theta = []
            for c, models in init_client_weights.items():
                for init_weights, new_weights in zip(models, client_model_weights[c]):
                    for k, p in init_weights.items():
                        hypernet_outputs.append(init_weights[k])
                        delta_theta.append(init_weights[k] - new_weights[k].to(self.device))

            # calculating phi gradient
            hnet_grads = torch.autograd.grad(hypernet_outputs, self.hypernet.parameters(), grad_outputs=delta_theta, allow_unused=True)

            # update hnet weights
            for p, g in zip(self.hypernet.parameters(), hnet_grads):
                p.grad = g
        else:
            grad_update_weights = []
            hnet_grads = []
            for c, models in init_client_weights.items():
                hypernet_outputs = []
                delta_theta = []
                for init_weights, new_weights in zip(models, client_model_weights[c]):
                    for k, p in init_weights.items():
                        hypernet_outputs.append(init_weights[k])
                        delta_theta.append(init_weights[k] - new_weights[k].to(self.device))

                # calculating phi gradient
                hnet_grads.append(torch.autograd.grad(hypernet_outputs, self.hypernet.parameters(), grad_outputs=delta_theta, allow_unused=True))
                grad_update_weights.append(client_update_weights[c])

            # update hnet weights
            for i, p in enumerate(self.hypernet.parameters()):
                g = torch.zeros_like(p)
                sum_update_weights = 0
                for c, c_grad in enumerate(hnet_grads):
                    if c_grad[i] is None:
                        continue
                    g += c_grad[i] * grad_update_weights[c]
                    sum_update_weights += grad_update_weights[c]

                if sum_update_weights == 0:
                    continue

                if p.grad is not None:
                    p.grad += g / sum_update_weights
                else:
                    p.grad = g / sum_update_weights

        nn.utils.clip_grad_norm_(self.hypernet.parameters(), 50)
        self.optimizer.step()


    def train(self, args):
        # types of messenge that server send to client
        # train: ask client to train model and return the model parameter
        # update: send the updated model to the client
        # stop: ask client to stop training and close connection
        if args.task == 'cifar10':
            unweighted = False
        else:
            unweighted = True

        self.logger.debug('---Start Registration---')
        threads = {}
        for cluster, cids in self.client_clusters.items():
            self.port = ((self.port - 1024) % (65535 - 1024)) + 1025
            send_msg = pickle.dumps({"subject": "register", "data": {"args": args, "ids": cids, "global_keys": self.global_keys}})

            socket_thread = SocketThread(
                addr=(self.ip, self.port),
                client_addr=cluster,
                send_msg=send_msg,
                buffer_size=args.buffer_size,
                timeout=self.timeout,
                logger=self.logger
            )
            socket_thread.start()
            threads[cluster] = socket_thread

        for cluster in threads:
            threads[cluster].join()
            self.client_features.update(threads[cluster].get_result()["client_features"])
        self.logger.debug('---Finish Registration---')

        self.all_selected_clients = set()
        for r in range(args.rounds):
            start_time = time.time()
            selected_clients = sorted(np.random.permutation(list(self.client_addr.keys()))[:args.sample_clients])
            self.all_selected_clients = self.all_selected_clients | set(selected_clients)
            init_client_weights = {}
            eval_client_weights = {}
            self.logger.critical(f'Round {r} selected clients: {selected_clients}')

            threads = {}
            for cluster in self.client_clusters:
                train_clients = [c for c in selected_clients if c in self.client_clusters[cluster]]
                eval_clients = self.client_clusters[cluster] - set(train_clients)

                # train_clients
                self.hypernet.train()
                init_cluster_weights = {c: self.hypernet(
                    torch.FloatTensor([self.client_features[c] for _ in self.client_models[c]]).to(self.device),
                    [deepcopy(model).to(self.device) for model in self.client_models[c]],
                    GraphBatch(self.graphs[c])) for c in train_clients}
                init_client_weights.update(init_cluster_weights)

                # eval_clients
                self.hypernet.eval()
                with torch.no_grad():
                    # calculate grad for regularization in server_update
                    eval_model_weights = {c: self.hypernet(
                        torch.FloatTensor([self.client_features[c]]).to(self.device),
                        [deepcopy(self.client_models[c][0]).to(self.device)], # only need the first model for evaluation
                        GraphBatch(self.graphs[c])) for c in eval_clients}
                    eval_client_weights.update(eval_model_weights)

                # model_weight - {global_key_idx: weight}
                send_msg = {"subject": "train_and_eval", "data": {
                    "round": r,
                    "train": {'ids': train_clients, "model": {c: [convert_model_key_to_idx(self.global_key_to_idx, model_name, init_cluster_weights[c][i]) for i, model_name in enumerate(self.client_model_names[c])] for c in train_clients}},
                    "eval": {"ids": eval_clients, "model": {c: [convert_model_key_to_idx(self.global_key_to_idx, model_name, eval_model_weights[c]) for i, model_name in enumerate(self.client_model_names[c])] for c in eval_clients}}
                }}

                # large device, send all other small models for distillation
                send_msg["data"]["global"] = [v for k, v in self.global_model_weights.items()]

                self.port = ((self.port - 1024) % (65535 - 1024)) + 1025

                socket_thread = SocketThread(
                    addr=(self.ip, self.port),
                    client_addr=cluster,
                    send_msg=pickle.dumps(send_msg),
                    buffer_size=args.buffer_size,
                    timeout=self.timeout,
                    logger=self.logger
                )
                socket_thread.start()
                threads[cluster] = socket_thread

            client_response = defaultdict(dict)
            for cluster in threads:
                threads[cluster].join()
                client_response.update(threads[cluster].get_result())
            update_client_weights = {c: res['model'] for c, res in client_response.items() if c in selected_clients}

            # store updated client weights
            for c in selected_clients:
                for i, model_name in enumerate(self.client_model_names[c]):
                    update_client_weights[c][i] = prepare_client_weights(self.client_models[c][i], model_name, {self.global_keys[k]: p for k, p in update_client_weights[c][i].items()})
                    missing_keys, unexpected_keys = self.client_models[c][i].load_state_dict(update_client_weights[c][i], strict=False)
                    if len(missing_keys) or len(unexpected_keys):
                        self.logger.debug('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))

            torch.cuda.empty_cache()

            self.logger.debug('Model Aggregation')
            self.server_update(init_client_weights, update_client_weights, {c: args.data_shares[c] for c in range(args.total_clients)}, unweighted=unweighted)
            updated_model_weights = self.aggregator.update(self.global_model_weights, update_client_weights, {c: args.data_shares[c] for c in range(args.total_clients)})
            self.global_model_weights.update(updated_model_weights)

            end_time = time.time()
            duration = (end_time - start_time) / 60.
            avg_scores = {'small': {}, 'large': {}}
            for metric in self.metrics:
                avg_scores['small'][metric] = np.average([client_response[c]['score'][metric] for c in range(args.total_clients - args.n_large)])
                avg_scores['large'][metric] = np.average([client_response[c]['score'][metric] for c in range(args.total_clients - args.n_large, args.total_clients)])
            self.logger.critical('[TRAIN] Round %i, time=%.3fmins, ACC-small=%.4f, ACC-large=%.4f' % (r, duration, avg_scores['small']['ACC'], avg_scores['large']['ACC']))
            for c in client_response:
                self.logger.critical({c: {m: round(client_response[c]['score'][m], 4) for m in self.metrics}})


class SocketThread(threading.Thread):
    def __init__(self, addr, client_addr, send_msg, buffer_size=1024, timeout=10, logger=None):
        threading.Thread.__init__(self)
        self.addr = addr
        self.client_addr = client_addr
        self.send_msg = send_msg
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.logger = logger

    def run(self):
        try:
            self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            self.soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.soc.bind(self.addr)
            self.soc.connect(self.client_addr)
            self.logger.debug(f"Run a Thread for connection with {self.client_addr}. Send {round(len(self.send_msg) * 1e-9, 4)} Gb.")
            send(self.soc, self.send_msg, self.buffer_size)

            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = f"Waiting for data from {self.client_addr}. Starting at {time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} {time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec}"
            self.logger.debug(date_time)
            msg, status = recv(self.soc, self.buffer_size, self.timeout)
            self.received_data = msg["data"] # model weight
            self.logger.debug(f"Receive {msg['subject'].upper()} message from {self.client_addr}")
            if status == 0:
                self.logger.debug(f"Connection Closed with {self.client_addr} either due to inactivity for {self.timeout} sec or an error.")

        except BaseException as e:
            self.logger.error(f"Error Connecting to the Client {self.client_addr}: {e}")

        finally:
            self.soc.close()
            self.logger.debug(f'Close connection with {self.client_addr}.')

    def get_result(self):
        try:
            return self.received_data
        except Exception as e:
            self.logger.error(f"Error Getting Result from {self.client_addr}: {e}.")
            return None