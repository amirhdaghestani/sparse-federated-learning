""" Client Class """
import torch
import torch.nn as nn

from attack.attack import Attack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None  # Use GradScaler only for CUDA


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
        local_model = type(self.model)().to(device)
        local_model.load_state_dict(global_weights)
        local_model.train()

        is_under_attack = self.malicious and epoch >= self.attack_epoch
        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

        # Attack on Parameters
        if is_under_attack and self.attack_type in self.ATTACK_ON_PARAMETRS:
            global_weights_random = self.attack_func(global_weights=global_weights, **self.attack_args)
            local_model.load_state_dict(global_weights_random)

        for local_ep in range(self.local_epoch):
            total_loss = 0
            num_batches = 0

            for data, target in self.data_loader:
                data, target = data.to(device), target.to(device)

                # Attack on Data
                if is_under_attack and self.attack_type in self.ATTACK_ON_DATA:
                    # If the client is malicious and the current epoch >= attack_epoch, apply attack on input data
                    data, target = self.attack_func(data=data, target=target, **self.attack_args)

                with torch.amp.autocast(device_type=device.type):
                    output = local_model(data)
                    loss = nn.CrossEntropyLoss()(output, target)

                optimizer.zero_grad()
                if device.type == 'cuda':
                    # Use GradScaler for GPU
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard backward pass for CPU
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if not compute_gradient:
                    break

            if local_ep == self.local_epoch - 1 or not compute_gradient:
                avg_loss = total_loss / num_batches if return_avg_loss else None

            if not compute_gradient:
                break

        # Move only the state_dict to CPU
        state_dict_cpu = {key: value.cpu() for key, value in local_model.state_dict().items()}

        if return_params:
            # Compute parameter updates only for trainable parameters
            params = [
                (state_dict_cpu[key] - global_weights[key]) for key in global_weights.keys()
            ]

            # Attack on Gradient
            if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
                params = self.attack_func(grads=params, **self.attack_args)

            params = {key: params[i] for i, key in enumerate(global_weights.keys())}
        else:
            # Get the keys for trainable parameters only
            trainable_keys = [name for name, _ in local_model.named_parameters()]

            # Compute parameter updates only for trainable parameters
            params = [
                -1 * (state_dict_cpu[key] - global_weights[key]) / lr
                for key in trainable_keys
            ]

            # Attack on Gradient
            if is_under_attack and self.attack_type in self.ATTACK_ON_GRADIENT:
                params = self.attack_func(grads=params, **self.attack_args)

        del local_model

        return params, avg_loss