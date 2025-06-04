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

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
