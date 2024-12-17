""" Attack Schemes """
import torch

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
        return [torch.normal(mean=torch.mean(grad) + kwargs['lie_attack_factor'] * torch.std(grad), std=torch.std(grad), size=grad.shape).to(device) for grad in kwargs['grads']]

    def __call__(attack_func, *args, **kwargs):
        return attack_func(*args, **kwargs)
