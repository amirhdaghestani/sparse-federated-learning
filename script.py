import sys
import argparse
import wandb
import yaml
import torch

from model.model import DeeperCIFARCNN, AlexNet, SimpleCNNWithBatchNorm, PyTorchLeNet5, ThreeLayerFC
from server.server_sparse import SparseFLServer
from server.server_fedavg import FedAvgServer

# Mapping of model names to classes
MODEL_MAP = {
    "DeeperCIFARCNN": DeeperCIFARCNN,
    "AlexNet": AlexNet,
    "SimpleCNNWithBatchNorm": SimpleCNNWithBatchNorm,
    "PyTorchLeNet5": PyTorchLeNet5,
    "ThreeLayerFC": ThreeLayerFC,
}

def load_config_from_yaml(filepath):
    """Load configuration from YAML."""
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def write_config_to_yaml(filepath, config):
    """Write configuration to YAML."""
    with open(filepath, 'w') as file:
        yaml.safe_dump(config, file, sort_keys=False)

def get_model(model_name):
    """Retrieve model instance by name."""
    if model_name in MODEL_MAP:
        return MODEL_MAP[model_name]()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def train(config, model):
    """Train model based on configuration."""
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
        sparse_params = config.get("sparse_params", {})
        server = SparseFLServer(**server_args)
        server.run(**sparse_params)
    elif aggregate_type == "fedavg":
        fedavg_params = config.get("fedavg_params", {})
        alpha = fedavg_params.get("alpha")
        if alpha is None:
            raise ValueError("FedAvgServer.run() requires 'alpha' parameter.")
        server = FedAvgServer(**server_args)
        server.run(alpha=alpha)
    else:
        raise ValueError(f"Unknown aggregate_type: {aggregate_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Training and Sweeps")
    parser.add_argument("--config", type=str, help="Path to the .yaml configuration file")
    parser.add_argument("--write-config", type=str, help="Path to save the default configuration as .yaml")
    args = parser.parse_args()

    default_config = {
        "training_config": {
            "project_name": "federated_learning_project",
            "model_name": "DeeperCIFARCNN",
            "aggregate_type": "sparse",
            "dataset_name": "CIFAR10",
            "num_clients": 200,
            "fraction_malicious": 0.4,
            "total_epochs": 200,
            "q_factor": 0.8,
            "evaluate_each_epoch": 1,
            "attack_args": {"attack_type": "flip_labels", "attack_epoch": 5, "max_label": 9},
            "defence_args": {"defence_type": "no_defence"},
            "batch_size": 64,
            "local_epochs": 3,
            "malicious_type": "group_oriented",
            "sparse_params": {
                "alpha": 0.01,
                "beta": 1e-4,
                "is_ftotal": True,
                "lambda_val": [0, 0.0025, 100],
                "c_alpha": 1e-3,
                "rho_alpha": 0.5,
                "max_line_search_iterations_alpha": 0,
                "c_beta": 1e-3,
                "rho_beta": 0.5,
                "max_line_search_iterations_beta": 0,
            },
            "fedavg_params": {"alpha": 0.01},
        },
        "sweep_config": {
            "method": "bayes",
            "metric": {"name": "test_accuracy", "goal": "maximize"},
            "parameters": {
                "alpha": {"values": [0.025, 0.01, 0.006, 0.0015]},
                "beta": {"values": [8e-4, 6e-4, 4e-4, 2e-4]},
            },
        },
    }

    if args.write_config:
        write_config_to_yaml(args.write_config, default_config)
        print(f"Default configuration written to {args.write_config}")
        sys.exit()

    if not args.config:
        print("Error: --config must be provided unless --write-config is used.")
        sys.exit(1)

    config = load_config_from_yaml(args.config)

    if "sweep_config" in config and "training_config" in config:
        sweep_config = config["sweep_config"]
        training_config = config["training_config"]
        project_name = training_config["project_name"]

        def train_wrapper():
            # Initialize wandb before accessing wandb.config
            wandb.init(project=project_name, config=training_config)
            
            # Combine training_config and wandb.config
            combined_config = {**training_config, **dict(wandb.config)}
            
            # Get the model and start training
            model = get_model(combined_config.get("model_name", "DeeperCIFARCNN"))
            train(combined_config, model)

        # Start the sweep
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        wandb.agent(sweep_id, function=train_wrapper)
    elif "training_config" in config:
        training_config = config["training_config"]
        project_name = training_config["project_name"]

        model = get_model(training_config["model_name"])
        wandb.init(project=project_name, config=training_config)
        train(wandb.config, model)
    else:
        print("Invalid configuration file. Must contain both 'training_config' and 'sweep_config' for sweep mode.")
