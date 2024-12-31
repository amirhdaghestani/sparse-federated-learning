import wandb
import copy
import torch

from model.model import DeeperCIFARCNN, AlexNet, SimpleCNNWithBatchNorm, PyTorchLeNet5, ThreeLayerFC
from server.server import Server
from server.server_sparse import SparseFLServer
from server.server_fedavg import FedAvgServer


MODEL = DeeperCIFARCNN()


def train(model):
    # Hyperparameters will be taken from wandb.config
    config = wandb.config
    dataset_name = config.get("dataset_name", "MNIST")
    num_clients = config.get("num_clients", 10)
    fraction_malicious = config.get("fraction_malicious", 0.0)
    total_epochs = config.get("total_epochs", 10)
    q_factor = config.get("q_factor", 0.1)
    evaluate_each_epoch = config.get("evaluate_each_epoch", 1)
    attack_args = config.get("attack_args", None)
    defence_args = config.get("defence_args", None)
    aggregate_type = config.get("aggregate_type", "fedavg")
    batch_size = config.get("batch_size", 64)
    local_epochs = config.get("local_epochs", 1)
    malicious_type = config.get("malicious_type", "group_oriented")

    # Common arguments for both servers
    server_args = {
        "dataset_name": dataset_name,
        "num_clients": num_clients,
        "fraction_malicious": fraction_malicious,
        "attack_args": attack_args,
        "defence_args": defence_args,
        "total_epochs": total_epochs,
        "q_factor": q_factor,
        "model": model,
        "evaluate_each_epoch": evaluate_each_epoch,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "malicious_type": malicious_type,
    }

    if aggregate_type == "sparse":
        sparse_params = {
            "alpha": config.get("alpha", 1e-3),
            "beta": config.get("beta", 1e-4),
            "is_ftotal": True,
            "lambda_val": (0, config.get("lambda_max", 0.0025), config.get("lambda_end_epoch", 100)),
            "c_alpha": 1e-3,
            "rho_alpha": 0.5,
            "max_line_search_iterations_alpha": 0,
            "c_beta": 1e-3,
            "rho_beta": 0.5,
            "max_line_search_iterations_beta": 0,
        }
        server = SparseFLServer(**server_args)
        server.run(**sparse_params)
    elif aggregate_type == "fedavg":
        fedavg_params = {
            "alpha": config.get("alpha", 1e-3),
        }
        server = FedAvgServer(**server_args)
        server.run(**fedavg_params)
    else:
        raise ValueError(f"Unknown aggregate_type: {aggregate_type}")

# Sweep Configuration Sparse
sweep_config_sparse = {
    'method': 'bayes',  # Choose 'grid', 'random', or 'bayes'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'alpha': {'values': [0.025, 0.01, 0.006, 0.0015]},
        'beta': {'values': [1e-4, 5e-5, 2e-5]},
        'lambda_max': {'values': [0.0025, 0.004, 0.001]}
    }
}

# Sweep Configuration FedAvg
sweep_config_fedavg = {
    'method': 'bayes',  # Choose 'grid', 'random', or 'bayes'
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': {
        'alpha': {'values': [0.025, 0.01, 0.006, 0.0015]},
    }
}

if __name__ == "__main__":
    def train_wrapper():
        wandb.init(
            project="test",
            config={
                "aggregate_type": "sparse", # sparse or fedavg
                "dataset_name": "CIFAR10",
                "num_clients": 50,
                "fraction_malicious": 0.4,
                "total_epochs": 200,
                "alpha": 0.01,
                "beta": 1e-4,
                "q_factor": 0.6,
                "evaluate_each_epoch": 1,
                "attack_args": {
                    "attack_type" : "flip_labels",
                    "attack_epoch" : 2,
                    "max_label": 9
                },
                "defence_args": {
                    "defence_type" : "no_defence",
                },
                "lambda_max": 0.0025,
                "lambda_end_epoch": 100,
                "batch_size": 64,
                "local_epochs": 3,
                "malicious_type": "group_oriented"
            }
        )
        train(MODEL)

    train_wrapper()

    # To run for different hyperparameters
    # sweep_id = wandb.sweep(sweep_config_fedavg, project="federated_learning_sweep_fixed")
    # wandb.agent(sweep_id, function=train_wrapper)

