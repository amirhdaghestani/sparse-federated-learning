""" Attack Schemes """
import torch
from scipy.stats import norm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attack:
    # Function to flip labels for malicious clients

    # Attack on Data
    def flip_labels(*args, **kwargs):
        return kwargs['data'], max(kwargs['target'].tolist()) - kwargs['target']

    # Attack on Parameters
    def random_parameters(*args, **kwargs):
        return {name: torch.normal(mean=kwargs['random_parameters_mean'], std=kwargs['random_parameters_std'], size=param.shape).to(device)for name, param in kwargs['global_weights'].items()}

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
        num_clients = len(clients)

        malicious_gradients = [
            clients_grads[i]
            for i, client in enumerate(clients) if client.malicious
        ]

        s = num_clients // 2 + 1 - len(malicious_gradients)
        phi_value = (num_clients - s) / num_clients
        z = norm.ppf(phi_value)

        # Initialize dictionary to store crafted malicious gradient
        attacked_grad = {}

        if isinstance(clients_grads[0], dict):
            for_list = clients_grad[0].keys()
        else:
            for_list = range(len(clients_grads[0]))

        # Stack tensors for each key in the gradient dictionaries
        for key in for_list:
            # Extract tensors for the current key from all malicious gradients
            stacked_tensors = torch.stack([grad[key].float() for grad in malicious_gradients])
            mean_tensor = torch.mean(stacked_tensors, dim=0)
            std_tensor = torch.std(stacked_tensors, dim=0)

            # Craft the malicious gradient for the current key
            attacked_grad[key] = mean_tensor - z * std_tensor

        # Replace gradients for malicious clients
        for i, client in enumerate(clients):
            if client.malicious:
                clients_grads[i] = attacked_grad

        return clients_grads

    def __call__(attack_func, *args, **kwargs):
        return attack_func(*args, **kwargs)
