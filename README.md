# RecipFL

How Few Davids Improve One Goliath: Federated Learning in Resource-Skewed Edge Computing Environments. WWW 2024.

## Run Experiments
### Datasets
The datasets will be downloaded automatically to the `data` folder upon their first use.
To change the directory, adjust `--data_dir` in `run_client.py` accordingly.

### Environment
In our experiments, we use the following specific versions of Python libraries:
```
networkx==1.11
numpy==1.13.3
pandas==0.22.0
scikit_learn==1.4.0
torch==1.3.0
torch_summary==1.4.5
torchsummary==1.5.1
torchvision==0.4.1
tqdm==4.63.0
transformers==4.17.0
```
### Run
1. Launch client(s) to await requests from the server.
Specify the port by `--port` and the gpu device by `-g`.
Note that `total_clients` will be automatically allocated across connected ports, so it is possible to simulate all clients in a single process.
Upon execution, the command provides the client IP,  which is required as an argument (`--client_ip`) in the next step.
For example, to launch clients on port 8361 and GPU 6, run:
```
python run_client.py --port 8361 -g 6
```
2. Start the training on server. 
Specify the task by `-t`, the gpu device by `-g`, the client IP by `--client_ip` (replace `xxx.xxx.xx.xxx` with the IP obtained from the first step), the client ports by `--cp` (should match the ports used in the first step).
Configure other experiment hyperparameters as needed (e.g., `--distill`, `--scaling`. See `run_server.py` for details).
For example, to launch server training on CIFAR-10 using GPU 6, run:
```
python run_server.py -t cifar10 -g 6 --distill --scaling width --client_ip xxx.xxx.xx.xxx --cp 8361 --cp 8362
```

## Citation
Please cite the following paper if you found our framework useful. Thanks!
```
@inproceedings{zhang2024few,
  title={How Few Davids Improve One Goliath: Federated Learning in Resource-Skewed Edge Computing Environments},
  author={Zhang, Jiayun and Li, Shuheng and Huang, Haiyu and Wang, Zihan and Fu, Xiaohan and Hong, Dezhi and Gupta, Rajesh K and Shang, Jingbo},
  booktitle={Proceedings of the ACM on Web Conference 2024},
  pages={2976--2985},
  year={2024}
}
```
