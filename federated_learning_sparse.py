import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision import datasets, transforms


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
    
    def local_update(self, global_weights, epoch):
        """Client performs local update and sends gradients back to the server."""
        local_model = ThreeLayerFC().to(device)
        local_model.load_state_dict(global_weights)
        local_model.train()

        total_loss = 0
        num_batches = 0
        optimizer = optim.SGD(local_model.parameters(), lr=0.01)

        for data, target in self.data_loader:
            data, target = data.to(device), target.to(device)
            
            # If the client is malicious and the current epoch >= attack_epoch, apply label flipping
            if self.malicious and epoch >= self.attack_epoch:
                target = flip_labels(target)
            
            optimizer.zero_grad()
            output = local_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Collect gradients
        grads = [param.grad.clone() for param in local_model.parameters()]
        avg_loss = total_loss / num_batches
        
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
        client_loaders = [torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, idx), batch_size=64, shuffle=True) for idx in client_indices]

        # Select malicious clients
        num_malicious = int(fraction_malicious * num_clients)
        malicious_client_ids = random.sample(range(num_clients), num_malicious)
        
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

    def federated_learning(self):
        counter = -1
        for alpha, beta in [(0.1, 0.1)]:
            counter += 1
            self.list_m_next.append([])
            self.list_w_next.append([])
            num_clients = len(self.clients)
            # alpha = 0.1  # Learning rate
            # beta = 0.01 # Learning rate

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

            for epoch in range(self.total_epochs):
                if epoch == 8:
                    print("HEY!")
                print(f"Epoch {epoch+1}/{self.total_epochs}")
                client_losses = []
                client_gradients = []

                ######################################
                # Update the global model using G and w
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
                
                # Print the average loss across clients
                avg_loss = sum(F_T) / num_clients
                print(f"Average Loss: {avg_loss}")
                #######################################

                #######################################
                # Each client performs a local update and returns gradients
                global_weights = self.global_model.state_dict()
                for client in self.clients:
                    grads, avg_loss = client.local_update(global_weights, epoch)
                    client_gradients.append(grads)
                    client_losses.append(avg_loss)
                
                # Construct G (gradient matrix) and F_T (loss vector)
                G_next = client_gradients  # List of gradients from each client
                F_T_next = client_losses   # List of losses from each client

                # Calculate m_next for each client
                G_flat = self._flatten_tensors(G)
                G_next_flat = self._flatten_tensors(G_next)

                G_T_G_next = torch.matmul(G_flat.T, G_next_flat)
                G_T_G_next_w = torch.matmul(G_T_G_next, torch.tensor(w))
                m_next = (torch.tensor(w) + beta * alpha * G_T_G_next_w - beta * torch.tensor(F_T)).tolist()
                self.list_m_next[counter].append(m_next)
                z_next = sorted(m_next, reverse=True)
                index_order = np.flip(np.argsort(m_next))
                n_z = sum(e > 0 for e in z_next)
                w_tilde_next = [0] * num_clients
                w_next = [0] * num_clients
                for i in range(min(self.n_max, n_z)):
                    w_tilde_next[i] = z_next[i]
                    w_next[index_order[i]] = w_tilde_next[i]
                w_norm_one = sum(w_next)
                if w_norm_one != 0:
                    w_next_normalize = [e / w_norm_one for e in w_next]
                w_next_normalize = w_next
                self.list_w_next[counter].append(w_next_normalize)

                G = G_next
                F_T = F_T_next
                w = w_next_normalize
                #######################################

if __name__ == "__main__":
    # Initialize server and start federated learning
    num_clients = 5  # Total number of clients
    fraction_malicious = 0.2  # Fraction of malicious clients (e.g., 40%)
    attack_epoch = 0  # Malicious clients start label-flipping after this epoch
    total_epochs = 10  # Total number of epochs
    n_max = 4  # Maximum number of non malicious clients
    server = Server(num_clients=num_clients, fraction_malicious=fraction_malicious, attack_epoch=attack_epoch, total_epochs=total_epochs, n_max=n_max)
    server.federated_learning()