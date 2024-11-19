import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple Three-Layer Fully Connected Model
class ThreeLayerFC(nn.Module):
    def __init__(self):
        super(ThreeLayerFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to flip labels for malicious clients
def flip_labels(labels):
    return 9 - labels

class Client:
    def __init__(self, client_id, data_loader, malicious=False, attack_epoch=0):
        self.client_id = client_id
        self.data_loader = data_loader
        self.malicious = malicious
        self.attack_epoch = attack_epoch

    def local_update(self, global_weights, epoch, return_avg_loss=True, compute_gradient=True):
        local_model = ThreeLayerFC().to(device)
        local_model.load_state_dict(global_weights)
        local_model.train()

        total_loss = 0
        num_batches = 0

        for data, target in self.data_loader:
            data, target = data.to(device), target.to(device)

            # If the client is malicious and the current epoch >= attack_epoch, apply label flipping
            if self.malicious and epoch >= self.attack_epoch:
                target = flip_labels(target)

            output = local_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            if compute_gradient:
                loss.backward()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if return_avg_loss else None
        grads = [param.grad.clone() / num_batches for param in local_model.parameters()] if compute_gradient else None

        return grads, avg_loss

class Server:
    def __init__(self, num_clients, fraction_malicious, attack_epoch=0, total_epochs=5, q_factor=0.1):
        self.global_model = ThreeLayerFC().to(device)
        self.num_clients = 0
        self.clients = self._initialize_clients(num_clients, fraction_malicious, attack_epoch, q_factor)
        self.total_epochs = total_epochs
        self.list_m_next = []
        self.list_w_next = []

    def _initialize_clients(self, num_clients, fraction_malicious, attack_epoch, q_factor):
        self.num_clients = num_clients
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        num_label = max(train_dataset.targets.tolist()) + 1
        client_loaders = self._distribute_dataset(train_dataset, num_label, q_factor)

        num_malicious = int(fraction_malicious * num_clients)
        malicious_ids = random.sample(range(num_clients), num_malicious)
        print(f"Malicious Client Indices: {malicious_ids}")

        return [Client(i, client_loaders[i], malicious=(i in malicious_ids), attack_epoch=attack_epoch) for i in range(num_clients)]
    
    def _split(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        
    def _distribute_dataset(self, train_dataset, num_label, q_factor):
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
            group2client_idx.extend([num_group - 1] * (self.num_clients - len(group2client_idx)))
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
        
        client_loaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, client2data_idx[str(i)]), batch_size=128, shuffle=True) for i in range(self.num_clients)]

        return client_loaders

    def _flatten_tensors(self, input_list):
        flattened = [torch.cat([tensor.view(-1) for tensor in tensors]) for tensors in input_list]
        return torch.stack(flattened).T

    def federated_learning(self, alpha_vec, beta_vec, is_ftotal=True, lambda_val=(0, 0.05, None),
                           c_alpha=1e-4, rho_alpha=0.5, max_line_search_iterations_alpha=0,
                           c_beta=1e-2, rho_beta=0.5, max_line_search_iterations_beta=10):

        # Ensure alpha_vec and beta_vec are lists
        alpha_vec = [alpha_vec] if not isinstance(alpha_vec, list) else alpha_vec
        beta_vec = [beta_vec] if not isinstance(beta_vec, list) else beta_vec
        
        # Generate lambda_range
        num_steps = lambda_val[-1] if lambda_val[-1] else self.total_epochs
        lambda_range = np.linspace(lambda_val[0], lambda_val[1], num_steps).tolist()
        lambda_range += [lambda_val[1]] * max(0, self.total_epochs - len(lambda_range))

        for i, (alpha, beta) in enumerate(zip(alpha_vec, beta_vec)):
            self.list_m_next.append([])
            self.list_w_next.append([])
            num_clients = len(self.clients)

            # Initialize weights
            w = [1.0 / num_clients] * num_clients
            global_weights = self.global_model.state_dict()

            # Perform initial client updates to gather gradients and losses
            client_gradients, client_losses = self._gather_client_updates(global_weights, 0)
            G, F_T = client_gradients, client_losses
            G_next = G
            F_T_next = F_T

            for epoch in range(self.total_epochs):
                print(f"Epoch {epoch+1}/{self.total_epochs}")

                # Perform backtracking line search for alpha
                alpha = self._line_search_alpha(alpha, G, F_T, w, num_clients, c_alpha, rho_alpha, epoch, max_line_search_iterations_alpha)
                print(f"alpha: {alpha}")

                # Update global model using G and weights w
                params_copy = {key: val.clone() for key, val in self.global_model.state_dict().items()}
                self._theta_update(G, G_next, F_T_next, w, alpha, epoch)
                avg_loss_before_weight_update = sum(F_T_next) / num_clients

                # Update weights and gather new client updates
                avg_loss_after_weight_upadate, w = self._weight_update(G, G_next, F_T, w, beta, lambda_range[epoch], is_ftotal,
                                                                       max_line_search_iterations_beta, c_beta, rho_beta)

                self._theta_update(G, G_next, F_T_next, w, alpha, epoch, params_copy)

                print(f"Average Loss Before Weights Update: {avg_loss_before_weight_update}")
                print(f"Average Loss After Weights Update: {avg_loss_after_weight_upadate}")
                print(f"Sparse_Weight: {w}")

                # Update gradients and losses for the next epoch
                G, F_T = G_next, F_T_next

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

    def _weight_update(self, G, G_next, F_T_next, w, beta, lambda_value, is_ftotal, max_line_search_iterations, c_beta, rho_beta, eye_factor=1e-6):
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
        m_next = w_tensor + beta * G_T_G_next_w - beta * F_T_next_tensor if is_ftotal else w_tensor + beta * G_T_G_next_w
        w_next_normalize = self._sparse_projection_onto_simplex(m_next.tolist(), lambda_value)

        # Perform line search to optimize beta
        beta, w_next_normalize, m_next = self._line_search_for_beta(w_tensor, m_next, w_next_normalize, beta, G_T_G_next_w, F_T_next_tensor, 
                                                                    is_ftotal, lambda_value, max_line_search_iterations, c_beta, rho_beta)

        self.list_m_next[-1].append(m_next)
        self.list_w_next[-1].append(w_next_normalize)
        print(f"beta: {beta}")

        avg_loss = sum(F_T_next_tensor.tolist()) / num_clients
        return avg_loss, w_next_normalize

    def _line_search_for_beta(self, w_tensor, m_next, w_next_normalize, beta, G_T_G_next_w, F_T_next_tensor, is_ftotal, lambda_value, max_line_search_iterations, c_beta, rho_beta):
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
                m_next = w_tensor + beta * G_T_G_next_w - beta * F_T_next_tensor if is_ftotal else w_tensor + beta * G_T_G_next_w
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

if __name__ == "__main__":
    num_clients = 10
    fraction_malicious = 0.3
    attack_epoch = 6
    total_epochs = 30
    alpha_vec = [0.25]
    beta_vec = [0.025]
    q_factor = 0.9
    server = Server(num_clients, fraction_malicious, attack_epoch, total_epochs, q_factor)
    server.federated_learning(alpha_vec, beta_vec, is_ftotal=True, lambda_val=(0, 0.05, 20),
                              c_alpha=1e-4, rho_alpha=0.5, max_line_search_iterations_alpha=0,
                              c_beta=1e-2, rho_beta=0.5, max_line_search_iterations_beta=0)
