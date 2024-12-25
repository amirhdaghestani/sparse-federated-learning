import copy
import torch
import wandb
import numpy as np

from server.server_base import BaseServer
from defence.defence import Defence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FedAvgServer(BaseServer):
    """
    Implements the Federated Averaging strategy on top of the BaseServer.
    """

    def run(self, alpha):
        """
        Standard Federated Averaging procedure over self.total_epochs.
        """
        # Initialize local param updates
        global_weights = self.global_model.state_dict()

        client_params, _ = self._gather_client_updates(
            global_weights=global_weights,
            epoch=0,
            lr=alpha,
            compute_gradient=True,
            return_avg_loss=True,
            return_params=True
        )
        delta_local_weights = client_params

        # Training loop
        for epoch in range(self.total_epochs):
            print(f"FedAvg Epoch {epoch+1}/{self.total_epochs}")
            wandb.log({"fedavg_epoch": epoch+1})

            # Aggregate all client updates into the global model
            self._fed_avg_theta_update(delta_local_weights, alpha, epoch)

            # Evaluate periodically
            if epoch % self.evaluate_each_epoch == 0:
                test_acc, test_loss = self.calculate_accuracy(is_fedavg=True)
                wandb.log({
                    "fedavg_test_accuracy": test_acc,
                    "fedavg_test_loss": test_loss
                })

    def _aggeregate_params(self, delta_local_weights, eta=1):
        """
        Aggregates local parameter updates using a defense function 
        (if any) and updates self.global_model.
        """
        if self.defence_func is not None:
            aggregated = self.defence_func(
                delta_local_updates=delta_local_weights, 
                **(self.defence_args if self.defence_args else {})
            )
        else:
            # Simple average if no defense
            aggregated = self._no_defense_aggregate(delta_local_weights)

        # Combine aggregated deltas with the current global model
        global_weights = self.global_model.state_dict()
        for k in aggregated.keys():
            aggregated[k] = global_weights[k] + eta * aggregated[k]

        self.global_model.load_state_dict(aggregated)

    def _fed_avg_theta_update(self, delta_local_weights, alpha, epoch):
        """
        Uses the aggregated parameter updates from clients to update the 
        global model, then gathers new local updates.
        """
        # 1. Aggregate client deltas
        self._aggeregate_params(delta_local_weights)

        # 2. Gather new local updates for next iteration
        global_weights = self.global_model.state_dict()
        client_params, _ = self._gather_client_updates(
            global_weights=global_weights,
            epoch=epoch,
            lr=alpha,
            compute_gradient=True,
            return_avg_loss=True,
            return_params=True
        )
        # This replaces old local deltas with the new ones
        for i, param_dict in enumerate(client_params):
            delta_local_weights[i] = param_dict

    def _no_defense_aggregate(self, delta_local_updates):
        """
        If no defence mechanism is provided, do an equal average of the updates.
        """
        num_clients = len(delta_local_updates)
        keys = delta_local_updates[0].keys()

        aggregated = {k: torch.zeros_like(delta_local_updates[0][k]) for k in keys}
        for update in delta_local_updates:
            for k in keys:
                aggregated[k] += update[k] / num_clients
        return aggregated
