""" Server Class """
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import wandb

from model.model import SimpleCNNWithBatchNorm, PyTorchLeNet5, ThreeLayerFC
from client.client import Client

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Server:
    def __init__(self, dataset_name, num_clients, fraction_malicious, attack_args=None, total_epochs=5, q_factor=0.1, model=SimpleCNNWithBatchNorm(), evaluate_each_epoch=2):
        self.global_model = model.to(device)
        self.global_model_fedavg = copy.deepcopy(model).to(device)
        self.num_clients = 0
        self.test_dataset = None
        self.clients = self._initialize_clients(dataset_name, num_clients, model, fraction_malicious, attack_args, q_factor)
        self.total_epochs = total_epochs
        self.evaluate_each_epoch = evaluate_each_epoch
        self.list_m_next = []
        self.list_w_next = []

    def _initialize_clients(self, dataset_name, num_clients, model, fraction_malicious, attack_args, q_factor):
        self.num_clients = num_clients

        # Assign datasets
        self.train_dataset, self.test_dataset = self._load_dataset(dataset_name)

        # Use the targets directly from train_dataset
        num_label = max(self.train_dataset.targets.tolist()) + 1
        client_loaders = self._distribute_dataset(self.train_dataset, num_label, q_factor)

        num_malicious = int(fraction_malicious * num_clients)
        malicious_ids = random.sample(range(num_clients), num_malicious)
        print(f"Malicious Client Indices: {malicious_ids}")

        return [Client(i, model, client_loaders[i], malicious=(i in malicious_ids), attack_args=attack_args) for i in range(num_clients)]

    def _load_dataset(self, dataset_name):
        if dataset_name == "MNIST":
            transform = transforms.Compose([ 
                transforms.ToTensor(), 
                transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std 
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        elif dataset_name == "EMNIST":
            transform = transforms.Compose([transforms.ToTensor()])
            split = 'balanced'
            train_dataset = datasets.EMNIST('./data', train=True, split=split, download=True, transform=transform)
            test_dataset = datasets.EMNIST('./data', train=False, split=split, download=True, transform=transform)

        return train_dataset, test_dataset

    def _split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        
    def _distribute_dataset(self, train_dataset, num_label, q_factor):
        if self.num_clients < num_label:
            raise Exception("Number of clients should be greater than the number of classes")

        num_group = num_label

        # Indice Labels
        label2idx = {}
        for i in range(num_label):
            label2idx[str(i)] = np.where(train_dataset.targets == i)[0]

        # Indice Group
        group2client_idx = []
        for i in range(num_group):
            group2client_idx.extend([i] * int(self.num_clients/num_group))
        if len(group2client_idx) < self.num_clients:
            np.random.permutation(num_group)
            group2client_idx.extend(np.random.permutation(num_group).tolist()[:(self.num_clients - len(group2client_idx))])
        random.shuffle(group2client_idx)

        # Construct Group to Data Index
        group2data_idx = []
        for i in range(num_group):
            group2data_idx.append([])

        for i in range(num_group):
            perm_idx = np.random.permutation(len(label2idx[str(i)]))
            q_group_idx, q_group_compliment_idx = perm_idx[:int(len(perm_idx) * q_factor)], perm_idx[int(len(perm_idx) * q_factor):]
            group2data_idx[i].extend(label2idx[str(i)][q_group_idx.tolist()])

            if len(q_group_compliment_idx) > 0:
                q_group_compliment_per_rest_idx = [e for e in self._split(q_group_compliment_idx, num_group - 1)]
                counter = 0
                for j in range(num_group):
                    if j != i:
                        group2data_idx[j].extend(label2idx[str(i)][q_group_compliment_per_rest_idx[counter].tolist()])
                        counter += 1

        for sublist in group2data_idx:
            random.shuffle(sublist) 

        # Construct Clinet to Data Index
        client2data_idx = {}
        for i in range(num_group):
            client_per_group_idx = np.where(np.array(group2client_idx) == i)[0]
            num_client_per_group = sum(np.array(group2client_idx) == i)
            indices = group2data_idx[i]
            list_clients = [e for e in self._split(indices, num_client_per_group)]
            for j, c in enumerate(client_per_group_idx):
                client2data_idx[str(c)] = list_clients[j]
        
        client_loaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, client2data_idx[str(i)]), batch_size=64, shuffle=True) for i in range(self.num_clients)]

        return client_loaders

    def _flatten_tensors(self, input_list):
        flattened = [torch.cat([tensor.view(-1) for tensor in tensors]) for tensors in input_list]
        return torch.stack(flattened).T

    def fed_avg(self, alpha):
        num_clients = len(self.clients)

        # Initialize weights
        global_weights = self.global_model_fedavg.state_dict()
        client_gradients, _ = self._gather_client_updates(global_weights, 0, True, True)
        G = client_gradients

        test_acc_list = []
        test_loss_list = []

        for epoch in range(self.total_epochs):
            print(f"FedAvg Epoch {epoch+1}/{self.total_epochs}")
            wandb.log({"fedavg_epoch": epoch+1})
            # Train local models and recieve gradients of clients
            # Update global weights = global weights - alpha * W * (clients_model)
            self._fed_avg_theta_update(G, alpha, epoch)
            # Calc acc every epoch
            if epoch % self.evaluate_each_epoch == 0:
                test_acc, test_loss = self.calculate_accuracy(is_fedavg=True)
                wandb.log({
                    "fedavg_test_accuracy": test_acc,
                    "fedavg_test_loss": test_loss
                })

                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)

    def _fed_avg_theta_update(self, G, alpha, epoch, params_copy=None):
        """Performs theta updates for the global model."""
        num_clients = len(self.clients)
        with torch.no_grad():
            for param_idx, (name, param) in enumerate(self.global_model_fedavg.named_parameters()):
                # Initialize the aggregated gradient
                agg_grad = torch.zeros_like(param.data)
                # Aggregate gradients weighted by w
                for client_idx in range(num_clients):
                    grad = G[client_idx][param_idx]
                    agg_grad += 1/num_clients * grad
                param.data =  param.data - alpha * agg_grad if params_copy is None else params_copy[name].data - alpha * agg_grad

        global_weights = self.global_model_fedavg.state_dict()
        client_gradients, _ = self._gather_client_updates(global_weights, epoch, True, True)

        G[:] = client_gradients

    def sparse_federated_learning(self, alpha, beta, is_ftotal=True, lambda_val=(0, 0.05, None),
                                  c_alpha=1e-4, rho_alpha=0.5, max_line_search_iterations_alpha=0,
                                  c_beta=1e-2, rho_beta=0.5, max_line_search_iterations_beta=10):

        # Generate lambda_range
        num_steps = lambda_val[-1] if lambda_val[-1] else self.total_epochs
        lambda_range = np.linspace(lambda_val[0], lambda_val[1], num_steps).tolist()
        lambda_range += [lambda_val[1]] * max(0, self.total_epochs - len(lambda_range))

        num_clients = len(self.clients)

        # Initialize weights
        w = [1.0 / num_clients] * num_clients
        global_weights = self.global_model.state_dict()

        # Perform initial client updates to gather gradients and losses
        client_gradients, client_losses = self._gather_client_updates(global_weights, 0)
        G, F_T = client_gradients, client_losses
        G_next = G
        F_T_next = F_T

        test_acc_list, test_loss_list = [], []

        for epoch in range(self.total_epochs):
            print(f"Epoch {epoch+1}/{self.total_epochs}")
            wandb.log({"epoch": epoch+1})

            # Perform backtracking line search for alpha
            alpha = self._line_search_alpha(alpha, G, F_T_next, w, num_clients, c_alpha, rho_alpha, epoch, max_line_search_iterations_alpha)
            print(f"alpha: {alpha}")
            wandb.log({"alpha": alpha})

            # Update global model using G and weights w
            params_copy = {key: val.clone() for key, val in self.global_model.state_dict().items()}
            self._theta_update(G, G_next, F_T_next, w, alpha, epoch)
            avg_loss_before_weight_update = np.matmul(np.transpose(np.array(F_T_next)), np.array(w))

            # Update weights and gather new client updates
            w = self._weight_update(G, G_next, F_T_next, w, alpha, beta, lambda_range[epoch], is_ftotal,
                                    max_line_search_iterations_beta, c_beta, rho_beta)

            self._theta_update(G, G_next, F_T_next, w, alpha, epoch, params_copy)
            avg_loss_after_weight_update = np.matmul(np.transpose(np.array(F_T_next)), np.array(w))

            print(f"Average Loss Before Weights Update: {avg_loss_before_weight_update}")
            print(f"Average Loss After Weights Update: {avg_loss_after_weight_update}")
            print(f"Sparse_Weight: {w}")

            wandb.log({
                "avg_loss_before_weight_update": float(avg_loss_before_weight_update),
                "avg_loss_after_weight_update": float(avg_loss_after_weight_update),
                "lambda_current": lambda_range[epoch],
                "beta": float(beta)
            })

            # Update gradients and losses for the next epoch
            G = G_next

            if epoch % self.evaluate_each_epoch == 0:
                test_acc, test_loss = self.calculate_accuracy()
                wandb.log({
                    "test_accuracy": test_acc,
                    "test_loss": test_loss
                })

                test_acc_list.append(test_acc)
                test_loss_list.append(test_loss)

        return test_acc_list, test_loss_list

    def _gather_client_updates(self, global_weights, epoch, return_avg_loss=True, compute_gradient=True):
        """Gathers initial client gradients and losses."""
        client_gradients = []
        client_losses = []
        for client in self.clients:
            grads, avg_loss = client.local_update(global_weights, epoch, return_avg_loss, compute_gradient)
            client_gradients.append(grads)
            client_losses.append(avg_loss)
        return client_gradients, client_losses

    def _line_search_alpha(self, alpha, G, F_T, w, num_clients, c, rho, epoch, max_iteration=3):
        """Performs line search for alpha using Armijo condition."""
        params_new = {key: val.clone() for key, val in self.global_model.state_dict().items()}
        for _ in range(max_iteration):  # Maximum iterations for line search
            agg_grad_vector = []
            with torch.no_grad():
                for param_idx, (name, param) in enumerate(self.global_model.named_parameters()):
                    # Initialize the aggregated gradient
                    agg_grad = torch.zeros_like(param.data)
                    # Aggregate gradients weighted by w
                    for client_idx in range(num_clients):
                        grad = G[client_idx][param_idx]
                        agg_grad += w[client_idx] * grad
                    params_new[name] = param - alpha * agg_grad  # Update rule
                    agg_grad_vector.append(agg_grad)

            # Evaluate new loss
            new_client_losses = self._gather_client_updates(params_new, epoch, return_avg_loss=True, compute_gradient=False)[1]
            L_new, L_old = np.mean(new_client_losses), np.mean(F_T)
            agg_grad_tensor = torch.cat([tensor.view(-1) for tensor in agg_grad_vector])

            if L_new <= L_old - c * alpha * torch.norm(agg_grad_tensor) ** 2:
                return alpha
            alpha *= rho
        if max_iteration != 0:
            print("Line search for alpha failed.")
        return alpha

    def _theta_update(self, G, G_next, F_T_next, w, alpha, epoch, params_copy=None):
        """Performs theta updates for the global model."""
        num_clients = len(self.clients)
        with torch.no_grad():
            for param_idx, (name, param) in enumerate(self.global_model.named_parameters()):
                # Initialize the aggregated gradient
                agg_grad = torch.zeros_like(param.data)
                # Aggregate gradients weighted by w
                for client_idx in range(num_clients):
                    grad = G[client_idx][param_idx]
                    agg_grad += w[client_idx] * grad
                param.data =  param.data - alpha * agg_grad if params_copy is None else params_copy[name].data - alpha * agg_grad

        global_weights = self.global_model.state_dict()
        client_gradients, client_losses = self._gather_client_updates(global_weights, epoch, True, True)

        G_next[:] = client_gradients
        F_T_next[:] = client_losses

    def _weight_update(self, G, G_next, F_T_next, w, alpha, beta, lambda_value, is_ftotal, max_line_search_iterations, c_beta, rho_beta, eye_factor=1e-6):
        """Performs weight updates with backtracking line search for w."""
        num_clients = len(self.clients)
        G_flat = self._flatten_tensors(G)
        G_next_flat = self._flatten_tensors(G_next)
        w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        F_T_next_tensor = torch.tensor(F_T_next, dtype=torch.float32, device=device)
        G_T_G_next = torch.matmul(G_flat.T, G_next_flat)
        G_T_G_next += eye_factor * torch.eye(G_T_G_next.shape[0])
        G_T_G_next_w = torch.matmul(G_T_G_next, w_tensor)

        # Initial computation of m_next
        m_next = w_tensor + alpha * beta * G_T_G_next_w - beta * F_T_next_tensor if is_ftotal else w_tensor + alpha * beta * G_T_G_next_w
        w_next_normalize = self._sparse_projection_onto_simplex(m_next.tolist(), lambda_value)

        # Perform line search to optimize beta
        beta, w_next_normalize, m_next = self._line_search_for_beta(w_tensor, m_next, w_next_normalize, alpha, beta, G_T_G_next_w, F_T_next_tensor, 
                                                                    is_ftotal, lambda_value, max_line_search_iterations, c_beta, rho_beta)

        self.list_m_next.append(m_next)
        self.list_w_next.append(w_next_normalize)
        print(f"beta: {beta}")

        return w_next_normalize

    def _line_search_for_beta(self, w_tensor, m_next, w_next_normalize, alpha, beta, G_T_G_next_w, F_T_next_tensor, is_ftotal, lambda_value, max_line_search_iterations, c_beta, rho_beta):
        """Performs backtracking line search for beta to optimize the weight update."""
        for _ in range(max_line_search_iterations):  # Maximum iterations for line search
            # Compute criterion for Armijo condition
            f_new = 0.5 * torch.norm(torch.tensor(w_next_normalize) - m_next) ** 2
            f_current = 0.5 * torch.norm(w_tensor - m_next) ** 2
            grad_f_T = (w_tensor - m_next).T
            armijo_condition = f_new <= f_current + c_beta * beta * grad_f_T @ (torch.tensor(w_next_normalize) - w_tensor)

            if armijo_condition:
                break
            else:
                beta *= rho_beta
                # Recompute m_next with the updated beta
                m_next = w_tensor + alpha * beta * G_T_G_next_w - beta * F_T_next_tensor if is_ftotal else w_tensor + alpha * beta * G_T_G_next_w
                w_next_normalize = self._sparse_projection_onto_simplex(m_next.tolist(), lambda_value)
        else:
            if max_line_search_iterations != 0:
                print("Line search for beta failed.")

        return beta, w_next_normalize, m_next

    def _sparse_projection_onto_simplex(self, m_next, lambda_value):
        # Sort m_next in descending order
        sorted_m = np.sort(m_next)[::-1]
        indices = np.argsort(m_next)[::-1]

        # Identify elements greater than lambda_value
        valid_indices = np.abs(sorted_m) > lambda_value
        if not np.any(valid_indices):
            return [0] * len(m_next)

        # Compute the cumulative sum of valid elements
        P_L_lambda = sorted_m[valid_indices]
        cumulative_sum = np.cumsum(P_L_lambda)
        rho_candidates = (P_L_lambda > (cumulative_sum - 1) / np.arange(1, len(P_L_lambda) + 1))

        # Determine the value of rho and etha
        if np.any(rho_candidates):
            rho = np.max(np.where(rho_candidates)[0]) + 1
            etha = (cumulative_sum[rho - 1] - 1) / rho
        else:
            etha = cumulative_sum[-1] / len(P_L_lambda)

        # Compute the projection onto the simplex
        P_plus = np.maximum(P_L_lambda - etha, 0)
        projected_w = np.zeros(len(m_next))
        projected_w[indices[valid_indices]] = P_plus

        return projected_w.tolist()

    def calculate_accuracy(self, is_fedavg=False):
        def acc(dataset, is_fedavg):
            correct = 0
            total_samples = 0
            total_batch = 0
            total_loss = 0

            if is_fedavg:
                model = self.global_model_fedavg
            else:
                model = self.global_model

            # Ensure model is in evaluation mode
            model.eval()

            # Disable gradient calculations for evaluation
            with torch.no_grad():
                for inputs, labels in torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False):
                    # Move data to the same device as the model
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Get model predictions
                    output = model(inputs)
                    loss = nn.CrossEntropyLoss()(output, labels)
                    predicted = torch.argmax(output, dim=1)

                    # Calculate number of correct predictions
                    correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    total_batch += 1
                    total_loss += loss.item()

            # Compute accuracy
            accuracy = 100 * correct / total_samples
            avg_loss = total_loss / total_batch
            
            return accuracy, avg_loss

        test_acc, test_loss = acc(self.test_dataset, is_fedavg)

        print("Test Accuracy = {:.2f}%, Test Loss: {:.4f}".format(test_acc, test_loss))

        return test_acc, test_loss
