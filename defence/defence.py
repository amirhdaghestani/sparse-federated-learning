"""Class for defence"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Defence:

    def __init__(self, defence_args=None):
        defence_type = defence_args.get('defence_type', 'no_defense')

        if defence_type == "no_defense":
            self.func = self.no_defense
        elif defence_type == "krum":
            self.func = self.krum
        elif defence_type == "trimmed_mean":
            self.func = self.trimmed_mean
        elif defence_type == "bulyan":
            self.func = self.bulyan
        else:
            self.func = self.no_defense

    def no_defense(self, *args, **kwargs):
        delta_local_updates = kwargs["delta_local_updates"]
        num_clients = len(delta_local_updates)

         # Initialize averaged_weights with the same structure as the first update
        averaged_weights = {key: torch.zeros_like(delta_local_updates[0][key], dtype=torch.float32) for key in delta_local_updates[0].keys()}

        # Sum up weights from all clients
        for delta_local_weight in delta_local_updates:
            for key in averaged_weights.keys():
                averaged_weights[key] += delta_local_weight[key].to(torch.float32) / num_clients

        return averaged_weights

    def krum(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        non_malicious_count = kwargs.get('krum_factor', len(delta_local_updates) - 2)
        return_index = kwargs.get("return_index")

        num_updates = len(delta_local_updates)

        # Compute pairwise distances between updates efficiently on GPU
        keys = list(delta_local_updates[0].keys())
        updates_flat = torch.empty((num_updates, sum(delta_local_updates[0][key].numel() for key in keys)), device=device)

        for i, update in enumerate(delta_local_updates):
            updates_flat[i] = torch.cat([update[key].flatten().to(device) for key in keys])

        distances = torch.cdist(updates_flat, updates_flat, p=2) ** 2  # Compute pairwise distances on GPU

        # Compute scores for each update
        sorted_distances, _ = distances.sort(dim=1)
        scores = sorted_distances[:, 1:1 + non_malicious_count].sum(dim=1)  # Exclude self-distance and sum top-k distances

        # Select the update with the minimum score
        krum_index = scores.argmin().item()

        if return_index:
            return delta_local_updates[krum_index], krum_index

        # Return the selected update as the aggregated weights
        return delta_local_updates[krum_index]

    def trimmed_mean(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        beta = kwargs.get('trimmed_factor', 0.1)

        num_clients = len(delta_local_updates)
        trimmed_weights = {key: [] for key in delta_local_updates[0].keys()}

        for key in trimmed_weights.keys():
            for client_update in delta_local_updates:
                trimmed_weights[key].append(client_update[key])

            trimmed_weights[key] = torch.stack(trimmed_weights[key])
            sorted_weights, _ = torch.sort(trimmed_weights[key], dim=0)
            lower_bound = int(beta * num_clients)
            upper_bound = num_clients - lower_bound
            trimmed_weights[key] = sorted_weights[lower_bound:upper_bound].mean(dim=0)

        return trimmed_weights

    def bulyan(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        m = kwargs.get('bulyan_factor', len(delta_local_updates) // 4)
        n = len(delta_local_updates)

        # Step 1: Perform iterative Krum selection to create a candidate set
        candidate_set = []
        for _ in range(n - 2 * m):
            kwargs['delta_local_updates'] = delta_local_updates
            candidate, krum_index = self.krum(return_index=True, **kwargs)
            candidate_set.append(candidate)
            del delta_local_updates[krum_index]

        # Step 2: Perform trimmed mean aggregation over the candidate set
        aggregated_weights = {key: [] for key in candidate_set[0].keys()}

        for key in aggregated_weights.keys():
            for candidate in candidate_set:
                aggregated_weights[key].append(candidate[key])

            aggregated_weights[key] = torch.stack(aggregated_weights[key])
            sorted_weights, _ = torch.sort(aggregated_weights[key], dim=0)
            lower_bound = max(0, min(m, len(candidate_set) // 2))  # Ensure valid lower bound
            upper_bound = max(lower_bound + 1, len(candidate_set) - m)  # Ensure valid range

            aggregated_weights[key] = sorted_weights[lower_bound:upper_bound].float().mean(dim=0)

        return aggregated_weights

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)