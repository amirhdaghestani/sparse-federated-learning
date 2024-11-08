import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms
import cvxpy as cp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ThreeLayerFC model for simplicity
class ThreeLayerFC(nn.Module):
    def __init__(self):
        super(ThreeLayerFC, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

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
        """Client performs local update and sends gradients back to the server."""
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
        
        avg_loss = None
        grads = None

        if return_avg_loss:
            avg_loss = total_loss / num_batches

        if compute_gradient:
            # Collect gradients
            grads = [param.grad.clone() / num_batches for param in local_model.parameters()]

        return grads, avg_loss

class Server:
    def __init__(self, num_clients, fraction_malicious, n_max, attack_epoch=0, total_epochs=5):
        self.global_model = ThreeLayerFC().to(device)
        self.clients = []
        self.attack_epoch = attack_epoch
        self.total_epochs = total_epochs
        self.n_max = n_max
        self.list_m_next = []
        self.list_w_next = []


        # Load dataset and partition among clients
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        num_items = int(len(train_dataset)/num_clients)
        indices = np.random.permutation(len(train_dataset))
        client_indices = [indices[i*num_items:(i+1)*num_items] for i in range(num_clients)]
        client_loaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, idx), batch_size=128, shuffle=True) for idx in client_indices]

        # Select malicious clients
        num_malicious = int(fraction_malicious * num_clients)
        malicious_client_ids = random.sample(range(num_clients), num_malicious)
        print(f"Malicoius_index: {malicious_client_ids}")
        
        for i in range(num_clients):
            is_malicious = i in malicious_client_ids
            self.clients.append(Client(i, client_loaders[i], malicious=is_malicious, attack_epoch=self.attack_epoch))
    
    def _flatten_tensors(self, input_list):
        """
        Flatten tensors
        """
        output = []
        for col in input_list:
            flatten_tensor = [tensor.view(-1) for tensor in col]
            stacked_tensor = torch.concat(flatten_tensor)
            output.append(stacked_tensor)
        output = torch.stack(output).T

        return output

    def federated_learning(self, alpha_vec, beta_vec, is_ftotal=True, lambda_val=(0, 0.05, None), is_norm_one=True,
                           inner_iteration_range=3, alpha_decay_param=0.9, beta_decay_param=0.9):
        counter = -1
        if not isinstance(alpha_vec, list):
            alpha_vec = [alpha_vec]
        
        if not isinstance(beta_vec, list):
            beta_vec = [beta_vec]
        
        lambda_range = list(np.linspace(lambda_val[0], lambda_val[1], lambda_val[-1] if lambda_val[-1] else self.total_epochs))
        if len(lambda_range) != self.total_epochs:
            lambda_range.extend([lambda_val[1]] * (self.total_epochs - len(lambda_range)))

        for i in range(len(alpha_vec)):
            counter += 1
            self.list_m_next.append([])
            self.list_w_next.append([])
            num_clients = len(self.clients)

            # Initialize model parameters and weight vector w
            w = [1.0 / num_clients] * num_clients  # Equal weights initially
            global_weights = self.global_model.state_dict()

            client_losses = []
            client_gradients = []
            # Each client performs a local update and returns gradients
            for client in self.clients:
                grads, avg_loss = client.local_update(global_weights, 0)
                client_gradients.append(grads)
                client_losses.append(avg_loss)
                
            # Construct G (gradient matrix) and F_T (loss vector)
            G = client_gradients  # List of gradients from each client
            F_T = client_losses   # List of losses from each client

            G_next = G
            for epoch in range(self.total_epochs):
                beta = beta_vec[i]
                print(f"Epoch {epoch+1}/{self.total_epochs}")


                # Backtracking Line Search for alpha
                alpha = alpha_vec[i]
                params_new = {key: val.clone() for key, val in self.global_model.state_dict().items()} # Not sure about this.
                # Armijo condition
                c = 1e-4
                rho = 0.5
                for _ in range(0):
                    # Compute aggregated gradient
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

                    # Evaluate new loss without updating local models or computing gradients
                    new_client_losses = []
                    for client in self.clients:
                        _, avg_loss = client.local_update(params_new, epoch, return_avg_loss=True, compute_gradient=False)
                        new_client_losses.append(avg_loss)
                    L_new = np.mean(new_client_losses)
                    L_old = np.mean(F_T)

                    agg_grad_tensor = [tensor.view(-1) for tensor in agg_grad_vector]
                    agg_grad_tensor = torch.concat(agg_grad_tensor)

                    lhs = L_new
                    rhs = L_old - c * alpha * np.linalg.norm(agg_grad_tensor) ** 2  # agg_grad_vector is the flattened agg_grad
                    if lhs <= rhs:
                        break
                    else:
                        alpha *= rho
                else:
                    print("Line search for alpha failed to find a suitable step size.")
                    # You can decide whether to continue with current alpha or stop



                for inner_theta_iteration in range(inner_iteration_range):
                    ######################################
                    # Update the global model using G and w
                    client_losses = []
                    client_gradients = []
                    G = G_next # Not sure.
                    if( inner_theta_iteration + 1 == inner_iteration_range ):
                        params_copy = {key: val.clone() for key, val in self.global_model.state_dict().items()} # Not sure about this.
                    with torch.no_grad():
                        for param_idx, param in enumerate(self.global_model.parameters()):
                            # Initialize the aggregated gradient
                            agg_grad = torch.zeros_like(param.data)
                            # Aggregate gradients weighted by w
                            for client_idx in range(num_clients):
                                grad = G[client_idx][param_idx]
                                agg_grad += w[client_idx] * grad
                            # Update global parameters
                            param -= alpha * agg_grad  # Update rule
                    #######################################

                    #######################################
                    # Each client performs a local update and returns gradients
                    global_weights = self.global_model.state_dict()
                    for client in self.clients:
                        if inner_theta_iteration + 1 != inner_iteration_range:
                            grads, _ = client.local_update(global_weights, epoch, False)
                        else:
                            grads, avg_loss = client.local_update(global_weights, epoch, True)
                        client_losses.append(avg_loss)
                        client_gradients.append(grads)

                    # alpha *= alpha_decay_param
                    # Construct G (gradient matrix) and F_T (loss vector)
                    G_next = client_gradients  # List of gradients from each client
                F_T_next = client_losses   # List of losses from each client
                # Print the average loss across clients
                avg_loss = sum(F_T_next) / num_clients
                print(f"Average Loss After Theta Update: {avg_loss}")

                # alpha = alpha_vec[i] # Reset

                G_flat = self._flatten_tensors(G)
                for inner_w_iteration in range(inner_iteration_range):
                    # Calculate m_next for each client
                    G_next_flat = self._flatten_tensors(G_next)
                    w_tensor = torch.tensor(w).to(device).to(torch.float32)
                    F_T_next_tensor = torch.tensor(F_T_next).to(device).to(torch.float32)

                    G_T_G_next = torch.matmul(G_flat.T, G_next_flat)
                    # G_T_G_next += 1e-4 * torch.eye(G_T_G_next.shape[0])

                    G_T_G_next_w = torch.matmul(G_T_G_next, w_tensor)

                    c = 1e-2
                    rho = 0.5
                    for _ in range(1):
                        if is_ftotal:
                            m_next = (w_tensor + beta * alpha * G_T_G_next_w - beta * F_T_next_tensor).tolist()
                        else:
                            m_next = (w_tensor + beta * alpha * G_T_G_next_w).tolist()

                        # w_next_normalize = self._sparse_projection_onto_simplex(m_next, lambda_range[epoch])
                        # self.list_w_next[counter].append(w_next_normalize)
                        # Project onto simplex
                        w_next_normalize = self._sparse_projection_onto_simplex(m_next, lambda_range[epoch])

                        # Compute surrogate loss or criterion
                        # Since we may not have an explicit loss function for w, you might define a custom criterion
                        # For example, the norm of the change in w

                        f_new = 0.5 * np.linalg.norm(np.array(w_next_normalize) - np.array(m_next)) ** 2
                        f_current = 0.5 * np.linalg.norm(np.array(w) - np.array(m_next))**2

                        grad_f_T = np.transpose(np.array(w) - np.array(m_next))
                        armijo_condition = f_new <= f_current + c * beta * grad_f_T @ (np.array(w_next_normalize) - np.array(w))

                        # rhs = np.linalg.norm(np.array(w) - np.array(m_next)) ** 2 + c * beta * np.matmul((np.array(w) - np.array(m_next)).transpose(), np.array(np.array(w_next_normalize) - np.array(w)))
                        if armijo_condition:
                            break
                        else:
                            beta *= rho
                    else:
                        print("Line search for beta failed to find a suitable step size.")
                        # Decide whether to proceed or adjust beta differently

                    self.list_m_next[counter].append(m_next)
                    self.list_w_next[counter].append(w_next_normalize)


                    # Calculate G_next, F_T_next
                    w = w_next_normalize
                    with torch.no_grad():
                        for param_idx, (name, param) in enumerate(self.global_model.named_parameters()):
                            # Initialize the aggregated gradient
                            agg_grad = torch.zeros_like(param.data)
                            # Aggregate gradients weighted by w
                            for client_idx in range(num_clients):
                                grad = G[client_idx][param_idx]
                                agg_grad += w[client_idx] * grad
                            # Update global parameters
                            param.data = params_copy[name].data - alpha * agg_grad  # Update rule, Not sure!

                    client_losses = []
                    client_gradients = []
                    global_weights = self.global_model.state_dict()
                    for client in self.clients:
                        grads, avg_loss = client.local_update(global_weights, epoch, True)
                        client_losses.append(avg_loss)
                        client_gradients.append(grads)
                    G_next = client_gradients
                    F_T_next = client_losses
                    
                    # alpha *= alpha_decay_param
                    # beta *= beta_decay_param

                avg_loss = sum(F_T_next) / num_clients
                print(f"Average Loss After Weights Update: {avg_loss}")
                print(f"Sparse_Weight: {w}")
                G = G_next
                F_T = F_T_next
                #######################################

    def _norm_calculate(self, m_next, lambda_value, is_norm_one=True):
        index_order = [f for f, value in enumerate(m_next) if value > lambda_value]
        w_next = [0] * len(m_next)
        for i in range(len(index_order)):
            w_next[index_order[i]] = m_next[index_order[i]]
            if is_norm_one:
                w_next[index_order[i]] -= lambda_value
        return w_next

    def _sparse_projection_onto_simplex(self, m_next, lambda_value):
        w = sorted(m_next, reverse=True)
        index_order = np.flip(np.argsort(m_next))
        index_order_1 = [f for f, value in enumerate(w) if value > lambda_value]
        P_L_lambda = [w[i] for i in index_order_1]

        if len(P_L_lambda) == 0:
            return  [0] * len(w)

        cumulative_sum = np.cumsum(P_L_lambda)
        condition = P_L_lambda > (cumulative_sum - 1) / np.arange(1, len(P_L_lambda) + 1)
        if np.any(condition):
            rho = np.max(np.where(condition)[0] + 1)
            etha = 1 / rho * (cumulative_sum[rho - 1] - 1)
        else:
            # raise Exception("Rho is empty!")
            etha = cumulative_sum[-1] / len(P_L_lambda)

        P_L_lambda_ehta = P_L_lambda - etha
        P_plus = np.clip(P_L_lambda_ehta, 0, max(P_L_lambda_ehta)).tolist()
        beta_S = [0] * len(w)

        for i in range(len(index_order_1)):
            beta_S[index_order_1[i]] = P_plus[i]

        w_final = [0] * len(w)
        for i in range(len(index_order)):
            w_final[index_order[i]] = beta_S[i]

        return w_final

    def _normalize(self, w_next):
        w_norm_one = sum(w_next)
        if w_norm_one != 0:
            w_next_normalize = [e / w_norm_one for e in w_next]
        else:
            w_next_normalize = [1.0 / len(w_next)] * len(w_next)

        return w_next_normalize

if __name__ == "__main__":
    # Initialize server and start federated learning
    num_clients = 10  # Total number of clients
    fraction_malicious = 0.3  # Fraction of malicious clients (e.g., 40%)
    attack_epoch = 0  # Malicious clients start label-flipping after this epoch
    total_epochs = 30  # Total number of epochs
    n_max = 4  # Maximum number of non malicious clients
    alpha_vec = [0.25]  # Alpha vector
    beta_vec = [0.025]  # Beta vector
    server = Server(num_clients=num_clients, fraction_malicious=fraction_malicious, attack_epoch=attack_epoch, total_epochs=total_epochs, n_max=n_max)
    server.federated_learning(alpha_vec=alpha_vec, beta_vec=beta_vec, is_ftotal=True, lambda_val=(0, 0.05, 20), is_norm_one=True,
                              inner_iteration_range=1, alpha_decay_param=0.9, beta_decay_param=0.9)
