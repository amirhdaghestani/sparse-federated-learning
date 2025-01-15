import random
import copy
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model.model import SimpleCNNWithBatchNorm
from client.client import Client
from attack.attack import Attack
from defence.defence import Defence


device = torch.device("cpu")


class BaseServer:
    """
    Base Server class that handles common functionalities like:
      - Dataset loading
      - Client initialization
      - Dataset distribution
      - Gathering client updates
      - Accuracy calculation
    It does NOT implement a specific aggregation strategy.
    """

    ATTACK_ON_BENIGN_UPDATES = ['lie_attack']

    def __init__(
        self, 
        dataset_name,
        num_clients,
        fraction_malicious,
        attack_args=None,
        defence_args=None,
        total_epochs=5,
        q_factor=0.1,
        model=SimpleCNNWithBatchNorm(),
        evaluate_each_epoch=2,
        local_epochs=1,
        batch_size=64,
        malicious_type="group_oriented"
    ):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the dataset ('MNIST', 'EMNIST', etc.).
        num_clients : int
            Number of clients.
        fraction_malicious : float
            Fraction of clients that are malicious (0 to 1).
        attack_args : dict, optional
            Attack configuration.
        defence_args : dict, optional
            Defence configuration.
        total_epochs : int
            Number of global training rounds (epochs).
        q_factor : float
            Parameter for partial data sharing across classes.
        model : nn.Module, optional
            Model architecture.
        evaluate_each_epoch : int
            Frequency of evaluation on the test set.
        local_epochs : int
            Number of local epochs for each client.
        """
        # Assertion
        assert malicious_type in ["group_oriented", "random"]

        # Models
        self.global_model = copy.deepcopy(model).to(device)
        self.num_clients = 0
        self.test_dataset = None
        self.fraction_malicious = fraction_malicious

        # Training configuration
        self.total_epochs = total_epochs
        self.local_epochs = local_epochs
        self.evaluate_each_epoch = evaluate_each_epoch
        
        # Defense and Attack
        self.defence_args = defence_args
        self.defence_func = None
        if defence_args is not None:
            self.defence_func = Defence(defence_args=defence_args)

        self.attack_args = attack_args
        self.attack_func = None
        if attack_args is not None:
            self.attack_type = attack_args.get('attack_type', None)
            self.attack_epoch = attack_args.get('attack_epoch', -1)
            self.attack_func = Attack(attack_args=attack_args)

        # Initialize clients
        self.clients = self._initialize_clients(
            dataset_name=dataset_name,
            num_clients=num_clients,
            model=model,
            fraction_malicious=fraction_malicious,
            attack_args=attack_args,
            q_factor=q_factor,
            batch_size=batch_size,
            malicious_type=malicious_type
        )

        # For Sparse FL line-search
        self.list_m_next = []
        self.list_w_next = []

    def _initialize_clients(self, dataset_name, num_clients, model, fraction_malicious, attack_args, q_factor, batch_size, malicious_type):
        """
        Loads data, creates client data loaders, and marks some clients as malicious.
        """
        self.num_clients = num_clients
        self.train_dataset, self.test_dataset = self._load_dataset(dataset_name)

        # Number of classes = max label + 1
        num_label = max(self.train_dataset.targets.tolist()) + 1
        client_loaders, group2client_idx = self._distribute_dataset(self.train_dataset, num_label=num_label, q_factor=q_factor, batch_size=batch_size)

        if malicious_type == "random":
            num_malicious = int(fraction_malicious * num_clients)
            malicious_ids = random.sample(range(num_clients), num_malicious)
        elif malicious_type == "group_oriented":
            num_group_malicious = int(fraction_malicious * num_label)
            group_malicious_ids = random.sample(range(num_label), num_group_malicious)
            malicious_ids = []
            for group_malicious_id in group_malicious_ids:
                malicious_ids.extend(np.where(np.array(group2client_idx) == group_malicious_id)[0].tolist())
        print(f"Malicious Client Indices: {malicious_ids}")

        clients = []
        for i in range(num_clients):
            is_malicious = (i in malicious_ids)
            clients.append(Client(
                client_id=i,
                model=model,
                data_loader=client_loaders[i],
                malicious=is_malicious,
                attack_args=attack_args,
                local_epoch=self.local_epochs
            ))
        return clients

    def _load_dataset(self, dataset_name):
        """
        Loads the specified dataset.
        """
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(
                './data', train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                './data', train=False, download=True, transform=transform
            )
        elif dataset_name == "EMNIST":
            transform = transforms.Compose([transforms.ToTensor()])
            split = 'balanced'
            train_dataset = datasets.EMNIST(
                './data', train=True, split=split, 
                download=True, transform=transform
            )
            test_dataset = datasets.EMNIST(
                './data', train=False, split=split, 
                download=True, transform=transform
            )
        elif dataset_name == "CIFAR10": 
            # Common transformations for CIFAR-10 
            transform = transforms.Compose([ 
                # Data augmentation (optional) 
                transforms.ToTensor(), 
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                    (0.2470, 0.2435, 0.2616)) 
            ])
            train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform) 
            test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

            # Convert targets (labels) to a tensor
            train_dataset.targets = torch.tensor(train_dataset.targets)
            test_dataset.targets = torch.tensor(test_dataset.targets)

        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        return train_dataset, test_dataset

    def _split(self, arr, n):
        """
        Helper function: Splits arr into n parts as evenly as possible.
        """
        k, m = divmod(len(arr), n)
        return (arr[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n))

    def _distribute_dataset(self, train_dataset, num_label, q_factor, batch_size):
        """
        Distributes the dataset among clients in a partially overlapping manner 
        controlled by q_factor. 
        """
        if self.num_clients < num_label:
            raise ValueError("Number of clients must be >= number of classes.")

        label2idx = {}
        for i in range(num_label):
            label2idx[str(i)] = np.where(train_dataset.targets == i)[0]

        # Each label i is also one group i
        group2client_idx = []
        for i in range(num_label):
            group2client_idx.extend([i] * (self.num_clients // num_label))

        remainder = self.num_clients - len(group2client_idx)
        if remainder > 0:
            # Random assignment of leftover clients
            extra_groups = list(np.random.permutation(num_label)[:remainder])
            group2client_idx.extend(extra_groups)

        random.shuffle(group2client_idx)

        # Build group -> data index
        group2data_idx = [[] for _ in range(num_label)]
        for i in range(num_label):
            perm_idx = np.random.permutation(len(label2idx[str(i)]))
            split_point = int(len(perm_idx) * q_factor)
            # Indices for the group i
            q_group_idx = perm_idx[:split_point]
            q_complement_idx = perm_idx[split_point:]

            group2data_idx[i].extend(label2idx[str(i)][q_group_idx.tolist()])

            # Distribute complement among other groups
            if len(q_complement_idx) > 0:
                splitted_complement = list(self._split(q_complement_idx, num_label - 1))
                c_idx = 0
                for j in range(num_label):
                    if j != i:
                        group2data_idx[j].extend(label2idx[str(i)][splitted_complement[c_idx].tolist()])
                        c_idx += 1

        # Shuffle each group
        for sublist in group2data_idx:
            random.shuffle(sublist)

        # Each group has some subset of data. Now map group -> clients
        client2data_idx = {}
        for i in range(num_label):
            client_indices = np.where(np.array(group2client_idx) == i)[0]
            data_indices = group2data_idx[i]
            splitted_data = list(self._split(data_indices, len(client_indices)))
            for j, c in enumerate(client_indices):
                client2data_idx[c] = splitted_data[j]

        client_loaders = []
        for i in range(self.num_clients):
            subset_indices = client2data_idx[i]
            subset_ds = torch.utils.data.Subset(train_dataset, subset_indices)
            loader = torch.utils.data.DataLoader(subset_ds, batch_size=batch_size, shuffle=True)
            client_loaders.append(loader)

        return client_loaders, group2client_idx

    def _flatten_tensors(self, input_list):
        """
        Flattens each client's list/dict of tensors into a single 1D vector
        and stacks them into a 2D tensor (columns = clients).
        """
        flattened = []
        for tensors in input_list:
            # Each element in `input_list` is typically a list of grads or a dict
            if isinstance(tensors, dict):
                # Sort keys for consistency
                sorted_keys = sorted(tensors.keys())
                cat = torch.cat([tensors[k].view(-1) for k in sorted_keys])
            else:
                cat = torch.cat([t.view(-1) for t in tensors])
            flattened.append(cat)
        return torch.stack(flattened).T

    def _gather_client_updates(
        self,
        global_weights,
        epoch,
        lr,
        return_avg_loss=True,
        compute_gradient=True,
        return_params=False
    ):
        """
        Gathers updates (gradients or parameter deltas) and losses from all clients.
        Also applies any 'benign update' attacks (e.g., lie_attack) if configured.
        """
        client_gradients = []
        client_losses = []

        for client in self.clients:
            updates, avg_loss = client.local_update(
                global_weights=global_weights,
                epoch=epoch,
                return_avg_loss=return_avg_loss,
                compute_gradient=compute_gradient,
                return_params=return_params,
                lr=lr
            )
            client_gradients.append(updates)
            client_losses.append(avg_loss)

        torch.cuda.empty_cache()

        # Attack on benign updates
        if (
            self.attack_type in self.ATTACK_ON_BENIGN_UPDATES and 
            epoch >= self.attack_epoch and 
            self.attack_func is not None
        ):
            # Malicious manipulation of benign updates
            client_gradients, client_losses = self.attack_func(
                grads=client_gradients["grads"] if "grads" in client_gradients.keys() else client_gradients,
                losses=client_losses,
                clients=self.clients,
                **(self.attack_args if self.attack_args else {})
            )

        return client_gradients, client_losses

    def calculate_accuracy(self, is_fedavg=False):
        """
        Evaluates the model on the test dataset.
        For FedAvg, typically uses a separate global model 
        than for other strategies.
        """
        model = self.global_model
        model.eval()

        loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False)
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = loss_fn(out, y)

                _, preds = torch.max(out, dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                total_loss += loss.item()

        accuracy = 100.0 * total_correct / total_samples
        avg_loss = total_loss / len(loader)

        # Check for divergence
        if np.isnan(avg_loss) or avg_loss > 9:
            raise RuntimeError(f"Model diverged at epoch: Loss = {avg_loss}")

        print(f"Test Accuracy = {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss
