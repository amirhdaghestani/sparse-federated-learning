import wandb
import copy

from model.model import SimpleCNNWithBatchNorm, PyTorchLeNet5, ThreeLayerFC
from server.server import Server

MODEL = SimpleCNNWithBatchNorm()


def train(model):
    # Hyperparameters will be taken from wandb.config
    dataset_name = wandb.config.dataset_name
    num_clients = wandb.config.num_clients
    fraction_malicious = wandb.config.fraction_malicious
    total_epochs = wandb.config.total_epochs
    alpha = wandb.config.alpha
    beta = wandb.config.beta
    q_factor = wandb.config.q_factor
    evaluate_each_epoch = wandb.config.evaluate_each_epoch
    attack_args = wandb.config.attack_args

    model_copy = copy.deepcopy(model)
    server = Server(dataset_name, num_clients, fraction_malicious, attack_args, total_epochs, q_factor, model_copy, evaluate_each_epoch)
    # server.sparse_federated_learning(alpha, beta, is_ftotal=True, lambda_val=(0, wandb.config.lambda_max, wandb.config.lambda_end_epoch),
    #                                  c_alpha=1e-4, rho_alpha=0.5, max_line_search_iterations_alpha=0,
    #                                  c_beta=1e-2, rho_beta=0.5, max_line_search_iterations_beta=0)
    server.fed_avg(alpha)

# Sweep Configuration
sweep_config = {
    'method': 'grid',  # Choose 'grid', 'random', or 'bayes'
    'metric': {'name': 'Test Accuracy', 'goal': 'maximize'},
    'parameters': {
        'alpha': {'values': [0.02, 0.01, 0.008, 0.004]},
        'beta': {'values': [0.005, 0.001, 0.0005]},
        'lambda_max': {'values': [0.004, 0.005, 0.01, 0.02]}
    }
}

if __name__ == "__main__":
    def train_wrapper():
        wandb.init(project="test", config={
            "dataset_name": "MNIST",
            "num_clients": 50,
            "fraction_malicious": 0.2,
            "total_epochs": 5,
            "alpha": 0.0081,
            "beta": 0.001,
            "q_factor": 0.6,
            "evaluate_each_epoch": 1,
            "attack_args": {
                "attack_type" : "flip_labels",
                "attack_epoch" : 0
            },
            "lambda_max": 0.01,
            "lambda_end_epoch": 1
        })

        train(MODEL)

    train_wrapper()

    # sweep_id = wandb.sweep(sweep_config, project="federated_learning_sweep")
    # wandb.agent(sweep_id, function=train_wrapper)

