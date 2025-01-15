""" Attack Schemes """
import torch
from scipy.stats import norm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attack:
    def __init__(self, attack_args):
        attack_type = attack_args.get('attack_type', None)

        # Attack on Data
        if attack_type == 'flip_labels':
            self.func = self.flip_labels
        elif attack_type == 'backdoor':
            self.func = self.backdoor
        # Attack on Parameters
        elif attack_type == 'random_parameters':
            self.func = self.random_parameters
        # Attack on Gradient
        elif attack_type == 'boost_gradient':
            self.func = self.boost_gradient
        elif attack_type == 'gaussian_attack':
            self.func = self.gaussian_attack
        elif attack_type == 'gaussian_additive_attack':
            self.func = self.gaussian_additive_attack
        elif attack_type == 'lie_attack':
            self.func = self.lie_attack
        elif attack_type is not None:
            raise Exception("Attack type is invalid.")

    # Attack on Data
    def flip_labels(*args, **kwargs):
        return kwargs['data'], kwargs['max_label'] - kwargs['target']

    def backdoor(*args, **kwargs):
        backdoor_pattern = kwargs.get('backdoor_pattern', None)
        backdoor_target = kwargs.get('backdoor_target', None)
        if backdoor_pattern is None:
            raise ValueError("backdoor_pattern is required.")
    
        i, j, h, w, v = backdoor_pattern['i'], backdoor_pattern['j'], backdoor_pattern['h'], backdoor_pattern['w'], backdoor_pattern['v']
        C = kwargs['data'] .shape[1]
        
        # Apply backdoor to the specified region
        if isinstance(v, (int, float)):
            # Single value for all channels
            kwargs['data'] [:, :, i:i+h, j:j+w] = v
        elif isinstance(v, torch.Tensor) and v.shape == (C,):
            # Channel-specific values
            kwargs['data'] [:, :, i:i+h, j:j+w] = v.view(C, 1, 1)
        elif isinstance(v, list) and len(v) == C:
            # Channel-specific values (list)
            kwargs['data'] [:, :, i:i+h, j:j+w] = torch.tensor(v, device=device).view(C, 1, 1)
        else:
            raise ValueError("Invalid value for 'v'. Must be an int, float, or a tensor of shape [C].")
        
        if backdoor_target is not None:
            if backdoor_target == "random":
                kwargs['target'] = torch.randint(kwargs.get('min_label', 0), kwargs['max_label'], size=kwargs['target'].size(), device=device)
            else:
                kwargs['target'].fill_(backdoor_target)

        return kwargs['data'], kwargs['target'] # Number of channels (1 for MNIST, 3 for CIFAR)

    # Attack on Parameters
    def random_parameters(*args, **kwargs):
        mean_factor = kwargs.get('random_parameters_mean', 0)
        std_factor = kwargs.get('random_parameters_std', 2)
        global_weights = kwargs['global_weights']

        # Step 1: Store the original shapes of the tensors
        original_shapes = {name: param.shape for name, param in global_weights.items()}

        # Step 2: Flatten and concatenate all weights
        concatenated_weights = torch.cat([param.view(-1) for param in global_weights.values()]).to(device)

        # Step 3: Compute standard deviation and mean of concatenated parameters
        std_concat = torch.std(concatenated_weights)
        mean_concat = torch.mean(concatenated_weights)

        # Step 4: Add noise to the concatenated parameters
        additive_noise = torch.normal(
            mean=-1 * mean_factor * mean_concat,
            std=std_factor * std_concat,
            size=concatenated_weights.shape,
            device=device
        )
        noisy_weights = concatenated_weights + additive_noise

        # Step 5: Split the concatenated tensor back into original shapes
        split_sizes = [param.numel() for param in global_weights.values()]
        split_tensors = torch.split(noisy_weights, split_sizes)

        # Step 6: Reshape each split tensor to its original shape
        reshaped_tensors = {
            name: tensor.view(shape).to(device)
            for (name, shape), tensor in zip(original_shapes.items(), split_tensors)
        }

        return reshaped_tensors

    # Attack on Gradient
    def boost_gradient(*args, **kwargs):
        return [kwargs['boost_factor'] * grad for grad in kwargs['grads']]

    def gaussian_attack(*args, **kwargs):
        return [torch.normal(mean=kwargs['gaussian_attack_mean'], std=kwargs['gaussian_attack_std'], size=grad.shape).to(device) for grad in kwargs['grads']]

    def gaussian_additive_attack(*args, **kwargs):
        if kwargs['gaussian_additive_attack_is_split']:
            return [grad + torch.normal(mean=0, std=kwargs['gaussian_additive_attack_std_factor'] * torch.std(grad), size=grad.shape).to(device) for grad in kwargs['grads']]
        
        original_shapes = [grad.shape for grad in kwargs['grads']]
        grads_concat = torch.concat([grad.view(-1) for grad in kwargs['grads']]).to(device)
        std_concat = torch.std(grads_concat)
        additive_noise = torch.normal(mean=0, std=kwargs['gaussian_additive_attack_std_factor'] * std_concat, size=grads_concat.shape)
        grads_concat = grads_concat + additive_noise

        split_size = [torch.prod(torch.tensor(shape)).item() for shape in original_shapes]
        return [split.view(shape).to(device) for split, shape in zip(torch.split(grads_concat, split_size), original_shapes)]

    def lie_attack(*args, **kwargs):
        clients = kwargs['clients']
        clients_grads = kwargs['grads']
        clients_losses = kwargs.get('losses', None)
        num_clients = len(clients)

        clients_params = [clients_grads, clients_losses]

        malicious_gradients = [
            clients_grads[i]
            for i, client in enumerate(clients) if client.malicious
        ]

        malicious_losses = None
        if clients_losses is not None:
            malicious_losses = [
                clients_losses[i]
                for i, client in enumerate(clients) if client.malicious
            ]

        malicious_params = [malicious_gradients, malicious_losses]

        s = num_clients // 2 + 1 - len(malicious_gradients)
        phi_value = (num_clients - s) / num_clients
        z_default = norm.ppf(phi_value)

        z = kwargs.get("z", z_default)

        # Initialize dictionary to store crafted malicious gradient
        if isinstance(clients_grads[0], dict):
            for_list = clients_grads[0].keys()
            attacked_grad = {}
            attacked_losses = [[]] * len(clients_grads[0])
            attacked_params = [attacked_grad, attacked_losses]
            sign_z = -1
        else:
            for_list = range(len(clients_grads[0]))
            attacked_grad = [[]] * len(clients_grads[0])
            attacked_losses = [[]] * len(clients_grads[0])
            attacked_params = [attacked_grad, attacked_losses]
            sign_z = 1

        # Stack tensors for each key in the gradient dictionaries
        for k, malicious in enumerate(malicious_params):
            if malicious is not None:
                if isinstance(malicious[0], float):
                    mean_list = np.mean(malicious, axis=0)
                    std_list = np.std(malicious, axis=0)
                    attacked_params[k] = float(mean_list + sign_z * z * std_list)
                else:
                    for key in for_list:
                        # Extract tensors for the current key from all malicious gradients
                        stacked_tensors = torch.stack([grad[key].float() for grad in malicious])
                        mean_tensor = torch.mean(stacked_tensors, dim=0)
                        std_tensor = torch.std(stacked_tensors, dim=0)

                        # Craft the malicious gradient for the current key
                        attacked_params[k][key] = mean_tensor + sign_z * z * std_tensor

        # Replace gradients for malicious clients
        for i, client in enumerate(clients):
            if client.malicious:
                for j, param in enumerate(clients_params):
                    if param is not None:
                        param[i] = attacked_params[j]

        return clients_params

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
