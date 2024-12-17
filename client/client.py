""" Client Class """
import numpy as np
import torch
import torch.nn as nn
import copy

from attack.attack import Attack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Client:
    ATTACK_ON_DATA = ['flip_labels']
    ATTACK_ON_PARAMETRS = ['random_parameters']
    ATTACK_ON_GRADIENT = ['boost_gradient', 'gaussian_attack', 'gaussian_additive_attack', 'lie_attack']

    def __init__(self, client_id, model, data_loader, malicious=False, attack_args=None):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.malicious = malicious
        self.attack_args = attack_args

        if malicious and attack_args is None:
            raise Exception("attack_args is not provided.")

        if attack_args is not None:
            self.attack_type = attack_args['attack_type']
            self.attack_epoch = attack_args['attack_epoch']

            # Attack on Data
            if self.attack_type == 'flip_labels':
                self.attack_func = Attack.flip_labels
            # Attack on Parameters
            elif self.attack_type == 'random_parameters':
                self.attack_func = Attack.random_parameters
            # Attack on Gradient
            elif self.attack_type == 'boost_gradient':
                self.attack_func = Attack.boost_gradient
            elif self.attack_type == 'gaussian_attack':
                self.attack_func = Attack.gaussian_attack
            elif self.attack_type == 'gaussian_additive_attack':
                self.attack_func = Attack.gaussian_additive_attack
            elif self.attack_type == 'lie_attack':
                self.attack_func = Attack.lie_attack

    def local_update(self, global_weights, epoch, return_avg_loss=True, compute_gradient=True):
        local_model = copy.deepcopy(self.model.to(device))
        local_model.load_state_dict(global_weights)
        local_model.train()

        total_loss = 0
        num_batches = 0

        condition = self.malicious and epoch >= self.attack_epoch

        # Attack on Parameters
        if condition and self.attack_type in self.ATTACK_ON_PARAMETRS:
            global_weights_random = self.attack_func(global_weights=global_weights, **self.attack_args)
            local_model.load_state_dict(global_weights_random)

        for data, target in self.data_loader:
            data, target = data.to(device), target.to(device)

            # Attack on Gradients
            if condition and self.attack_type in self.ATTACK_ON_DATA:
                # If the client is malicious and the current epoch >= attack_epoch, apply attack on input data
                data, target = self.attack_func(data=data, target=target)

            output = local_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            if compute_gradient:
                loss.backward()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if return_avg_loss else None
        grads = [param.grad.clone() / num_batches for param in local_model.parameters()] if compute_gradient else None

        # Attack on Gradient
        if condition and self.attack_type in self.ATTACK_ON_GRADIENT:
            grads = self.attack_func(grads=grads, **self.attack_args)

        return grads, avg_loss