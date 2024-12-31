""" Client Class """
import numpy as np
import torch
import torch.nn as nn
import copy

from attack.attack import Attack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client:
    ATTACK_ON_DATA = ['flip_labels', 'backdoor']
    ATTACK_ON_PARAMETRS = ['random_parameters']
    ATTACK_ON_GRADIENT = ['boost_gradient', 'gaussian_attack', 'gaussian_additive_attack']

    def __init__(self, client_id, model, data_loader, local_epoch=1, malicious=False, attack_args=None):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.malicious = malicious
        self.local_epoch = local_epoch

        if malicious and attack_args is None:
            raise Exception("attack_args is not provided.")

        if attack_args is not None:
            self.attack_args = attack_args
            self.attack_type = attack_args['attack_type']
            self.attack_epoch = attack_args['attack_epoch']
            self.attack_func = Attack(attack_args)

    def local_update(self, global_weights, epoch, return_avg_loss=True, compute_gradient=True, return_params=False, lr=1e-3):
        local_model = copy.deepcopy(self.model.to(device))
        local_model.load_state_dict(global_weights)
        local_model.train()

        is_under_attack = self.malicious and epoch >= self.attack_epoch
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

        # Attack on Parameters
        if is_under_attack and self.attack_type in self.ATTACK_ON_PARAMETRS:
            global_weights_random = self.attack_func(global_weights=global_weights, **self.attack_args)
            local_model.load_state_dict(global_weights_random)

        total_grads = None
        for epoch in range(self.local_epoch):
            total_loss = 0
            num_batches = 0
            grad_trajectory = []

            for data, target in self.data_loader:
                data, target = data.to(device), target.to(device)

                # Attack on Data
                if is_under_attack and self.attack_type in self.ATTACK_ON_DATA:
                    # If the client is malicious and the current epoch >= attack_epoch, apply attack on input data
                    data, target = self.attack_func(data=data, target=target, **self.attack_args)

                output = local_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                if compute_gradient:
                    optimizer.zero_grad()
                    loss.backward()
                    ## Boost gradient Applied Here param.grad = - boost_factor * param.grad
                    ## Gaussian Attack Applied Here param.grad = random_normal
                    ## Gaussian Additive Noise param.grad += random_normal_additive_noise
                    ## Lie attack param.grad += random_noraml_additive_noise(std=scale_factor * std(param.grad))

                    # Attack on Gradient
                    grads = [param.grad.clone() for param in local_model.parameters()] if compute_gradient else None
                    if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
                        grads = self.attack_func(grads=grads, **self.attack_args)

                        # Apply modified gradients
                        for param, grad in zip(local_model.parameters(), grads):
                            param.grad = grad

                    # Initialize total_grads if it's the first batch
                    if total_grads is None:
                        total_grads = [torch.zeros_like(grad) for grad in grads]

                    for i, grad in enumerate(grads):
                        total_grads[i] += grad

                    optimizer.step()

                    # Remove temporarily
                    # if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
                    #     output = local_model(data)
                    #     loss = nn.CrossEntropyLoss()(output, target)

                total_loss += loss.item()
                num_batches += 1

            if epoch == self.local_epoch - 1:
                avg_loss = total_loss / num_batches if return_avg_loss else None

        grads = total_grads
        
        if return_params:
            params = {key: local_model.state_dict()[key] - global_weights[key] for key in global_weights.keys()}
        else:
            # Get the keys for trainable parameters only
            trainable_keys = [name for name, _ in local_model.named_parameters()]

            # Compute parameter updates only for trainable parameters
            params = [
                -1 * (local_model.state_dict()[key] - global_weights[key]) / lr
                for key in trainable_keys
            ]

        del local_model

        return params, avg_loss