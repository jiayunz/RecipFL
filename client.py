import os
import torch
from torch import nn
import numpy as np
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from datetime import datetime
import socket
from utils.communication_utils import send, recv
from utils.general_utils import set_seed
from tqdm import tqdm

from utils.evaluation import calculate_SLC_metrics, display_results
from utils.general_utils import prepare_client_weights, convert_model_key_to_idx
from utils.build_model import build_model

EPS = 1e-7

class ClientCluster():
    def __init__(self, port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()

        self.port = port
        self.server_ip = None
        self.clients = {}
        print('address:', (self.ip, self.port))

    def register_task(self, args, server_args, global_keys):
        self.global_keys = global_keys
        self.global_key_to_idx = {global_k: i for i, global_k in enumerate(self.global_keys)}
        args.task = server_args.task
        args.total_clients = server_args.total_clients
        args.n_large = server_args.n_large
        args.distill = server_args.distill
        args.classifier_name = server_args.classifier_name
        self.n_class = server_args.n_class
        args.finetune_epochs = server_args.finetune_epochs

        if server_args.task.startswith('cifar'):
            from utils.datasets.load_cifar import load_cifar
            trainData, valData, testData = load_cifar(server_args.task, os.path.join(args.data_dir, server_args.task), server_args.data_shares, server_args.alpha, server_args.n_large)
            collate_fn = None

        elif server_args.task == 'mnist':
            valData = [None] * server_args.total_clients
            from utils.datasets.load_mnist import load_mnist
            trainData, testData = load_mnist(os.path.join(args.data_dir, server_args.task), server_args.data_shares, server_args.alpha, server_args.n_large)
            collate_fn = None

        elif server_args.task == 'mnli':
            valData = [None] * server_args.total_clients
            from utils.datasets.load_mnli import load_mnli, collate_fn
            trainData, testData = load_mnli(os.path.join(args.data_dir, server_args.task, 'original'), server_args.data_shares, server_args.alpha, server_args.n_large)
            collate_fn = collate_fn

        else:
            raise ValueError('Wrong dataset.')

        return trainData, valData, testData, collate_fn


    def run(self, args):
        self.device = args.device
        # waiting for server to send request
        try:
            soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
            soc.bind((self.ip, self.port))
            soc.listen(1)
            print('Start Listening...')

            while True:
                try:
                    new_socket, source_addr = soc.accept()
                    new_socket.settimeout(args.timeout)
                    if self.server_ip is not None and source_addr[0] != self.server_ip:
                        new_socket.close()
                        print(f'\033[31mReceive Unexpected Connection from {source_addr}. Connection Close.\033[0m')

                    print(f'Receive connection from {source_addr}')
                    # receive request
                    msg, status = recv(new_socket, args.buffer_size, recv_timeout=60)
                    if status == 1:
                        print(f"Receive {msg['subject'].upper()} message from {source_addr}")

                    if isinstance(msg, dict):
                        if msg['subject'] == 'register':
                            self.server_ip = source_addr[0]
                            trainData, valData, testData, collate_fn = self.register_task(args, msg['data']['args'], msg['data']['global_keys'])
                            client_features = {}
                            for cid in msg['data']['ids']:
                                self.clients[cid] = Client(args, msg['data']['args'], cid, trainData[cid], valData[cid], testData[cid], collate_fn)
                                client_features[cid] = self.clients[cid].class_distribution

                            data_byte = pickle.dumps({"subject": "register", "data": {"client_features": client_features}})
                            print("Registered. Reply to the Server.")
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte

                        elif msg['subject'] == 'train_and_eval':
                            response_data = {}
                            # train
                            for cid in msg['data']['train']['ids']:
                                response_data[cid] = {"model": [], "score": None}
                                for i, model_name in enumerate(self.clients[cid].model_names):
                                    recv_weights = {self.global_keys[k_idx]: p for k_idx, p in msg['data']['train']['model'][cid][i].items()}
                                    new_weights = prepare_client_weights(self.clients[cid].models[i], model_name, recv_weights)
                                    # only pass teacher models to large device
                                    updated_weights, test_scores = self.clients[cid].local_update(args, i, msg['data']['round'], new_weights)
                                    response_data[cid]["model"].append(convert_model_key_to_idx(self.global_key_to_idx, model_name, updated_weights))
                                    display_results(test_scores, self.clients[cid].metrics)
                                    if i == 0:
                                        response_data[cid]["score"] = test_scores

                            # eval
                            for cid in msg['data']['eval']['ids']:
                                # don't update client model
                                model = deepcopy(self.clients[cid].models[0])
                                recv_weights = {self.global_keys[k_idx]: p for k_idx, p in msg['data']['eval']['model'][cid][0].items()}

                                new_weights = prepare_client_weights(self.clients[cid].models[0], self.clients[cid].model_names[0], recv_weights)
                                missing_keys, unexpected_keys = model.load_state_dict(new_weights, strict=False)
                                if len(missing_keys) or len(unexpected_keys):
                                    print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))
                                model = self.clients[cid].fine_tune(args, model)
                                test_scores, test_loss = self.clients[cid].evaluate(args, model)
                                print(f"Evaluated Client %i. Test loss = %.4f" % (cid, test_loss))
                                display_results(test_scores, self.clients[cid].metrics)
                                if cid in response_data:
                                    response_data[cid]["score"] = test_scores
                                else:
                                    response_data[cid] = {"score": test_scores}

                            # reply request
                            data_byte = pickle.dumps({"subject": "train_and_eval", "data": response_data})
                            print(f"Trained and evaluated. Send {len(data_byte)*1e-9} Gb to the Server.")
                            new_socket.settimeout(3600)
                            send(new_socket, data_byte, args.buffer_size)

                            del data_byte
                finally:
                    new_socket.close()
                    print(f'Close Connection with {source_addr}')
        finally:
            soc.close()


class Client():
    def __init__(self, args, server_args, id, trainData, valData, testData, collate_fn):
        self.id = id
        args.epochs = server_args.epochs
        args.buffer_size = server_args.buffer_size
        set_seed(server_args.seed)
        self.task = server_args.task
        self.is_large = id >= (server_args.total_clients - server_args.n_large)
        self.classifier_name = server_args.classifier_name
        self.device = args.device
        self.metrics = server_args.metrics

        self.trainData = trainData
        self.valData = valData
        self.testData = testData
        self.collate_fn = collate_fn

        self.n_class = server_args.n_class
        self.distill = server_args.distill

        # client features
        class_distribution = np.zeros(self.n_class)
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4)
        for _, labels in train_loader:
            for cls in range(self.n_class):
                class_distribution[cls] += labels.numpy().tolist().count(cls)

        self.class_distribution = class_distribution / np.sum(class_distribution)
        print(f'Client {id} class distribution:', self.class_distribution)
        self.model_names = server_args.client_model_names[id]
        self.models = []
        for model_name in self.model_names:
            model = build_model(model_name, self.task, self.n_class, self.device)
            if self.task == 'mnli':
                model.train_adapter("mnli")
            self.models.append(model)
        self.alpha = 0.01
        self.prev_grads = [0. for _ in self.model_names]

        print(f'Client {self.id} n_train: {len(self.trainData)}, n_class: {self.n_class}')


    def train_one_batch(self, model, model_idx, sample, label, optimizer):
        model.train()
        criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction="batchmean")

        label = label.to(self.device, dtype=torch.long)
        if len(label.shape) > 1:
            label = torch.argmax(label, dim=-1)

        optimizer.zero_grad()
        feat, out = self.model_fit(model.to(self.device), sample, return_emb=True)
        loss = criterion(out, label)

        # distillation
        if self.distill and len(self.models) > 1:
            if model_idx > 0:
                teacher = deepcopy(self.models[0].to(self.device))
                teacher.eval()
                t_feat, t_out = self.model_fit(teacher, sample, return_emb=True)

                logits_loss = criterion(out, t_out.softmax(dim=1))
                loss += logits_loss
                if t_feat.shape == feat.shape:
                    feature_loss = kl_criterion(F.log_softmax(feat, dim=1), F.softmax(t_feat, dim=1))
                    loss += feature_loss

        loss.backward(retain_graph=True)
        optimizer.step()
        return loss.item()

    def evaluate(self, args, model):
        data_loader = DataLoader(self.testData, batch_size=args.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=1)
        criterion = nn.CrossEntropyLoss()
        y_pred = []
        y_true = []

        model.eval()
        avg_loss = []
        with torch.no_grad():
            for sample, label in data_loader:
                label = label.to(self.device, dtype=torch.float)
                if len(label.shape) == 1:
                    label = F.one_hot(label.to(torch.long), num_classes=self.n_class)

                out = self.model_fit(model.to(self.device), sample)
                avg_loss.append(criterion(out, torch.argmax(label, dim=-1)).item())
                out = torch.softmax(out, dim=-1)

                y_pred.extend(out.cpu().numpy())
                y_true.extend(label.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        test_scores = calculate_SLC_metrics(y_true, y_pred)

        return test_scores, np.mean(avg_loss)

    def local_update(self, args, model_idx, round, model_weights):
        missing_keys, unexpected_keys = self.models[model_idx].load_state_dict(model_weights, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            print('Warning: missing %i missing_keys, %i unexpected_keys.' % (len(missing_keys), len(unexpected_keys)))
        self.recv_params = torch.cat([p.reshape(-1) for p in self.models[model_idx].to(self.device).parameters()])

        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.models[model_idx].parameters()), lr=args.lr)
        for e in range(args.epochs):
            start_time = datetime.now()
            for sample, label in tqdm(train_loader, total=len(train_loader)):
                self.train_one_batch(self.models[model_idx], model_idx, sample, label, optimizer)

            end_time = datetime.now()
            duration = (end_time - start_time).seconds / 60.
            print('[TRAIN] Client %i, Epoch %i, time=%.3fmins' % (self.id, round * args.epochs + e, duration))
            # client testing
            if e == args.finetune_epochs - 1: # test after fine_tune_epoch
                test_scores, _ = self.evaluate(args, self.models[model_idx])

        curr_params = torch.cat([p.reshape(-1) for p in self.models[model_idx].parameters()])
        self.prev_grads[model_idx] -= self.alpha * (curr_params.to(self.device) - self.recv_params.to(self.device))
        updated_weights = {k: p for k, p in self.models[model_idx].state_dict().items()}

        return updated_weights, test_scores

    # train model one round, without changing self.model value
    def fine_tune(self, args, model):
        self.recv_params = torch.cat([p.reshape(-1) for p in model.to(self.device).parameters()])
        train_loader = DataLoader(self.trainData, batch_size=args.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        model.train()

        for e in range(args.finetune_epochs):
            start_time = datetime.now()
            for sample, label in tqdm(train_loader, total=len(train_loader)):
                self.train_one_batch(model, 0, sample, label, optimizer)

            end_time = datetime.now()
            duration = (end_time - start_time).seconds / 60.
            print('[FINE-TUNE] Client %i, time=%.3fmins' % (self.id, duration))

        return model

    def model_fit(self, model, sample, return_emb=False):
        if self.task == 'mnli':
            output = model(sample[0].to(self.device), token_type_ids=sample[1].to(self.device), attention_mask=sample[2].to(self.device), output_hidden_states=True)
            if return_emb:
                return output.hidden_states[-1][:, 0], output.logits
            else:
                return output.logits
        else:
            return model(sample.to(self.device), return_emb=return_emb)