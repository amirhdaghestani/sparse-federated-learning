"""Class for defence"""
import torch


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
        delta_local_updates = kwargs['delta_local_updates']
        keys = delta_local_updates[0].keys()
        num_clients = len(delta_local_updates)

        # Incremental sum approach:
        aggregated = {
            k: torch.zeros_like(delta_local_updates[0][k]) for k in keys
        }

        for update in delta_local_updates:
            for k in keys:
                aggregated[k] += update[k]

        for k in keys:
            aggregated[k] /= num_clients

        return aggregated

    def krum(self, *args, **kwargs):
        delta_local_updates = kwargs['delta_local_updates']
        non_malicious_count = kwargs.get('krum_factor', len(delta_local_updates) - 2)
        return_index = kwargs.get("return_index", False)

        # Infer the device from the first tensor in delta_local_updates
        device = next(iter(delta_local_updates[0].values())).device

        num_updates = len(delta_local_updates)
        if num_updates < 2:
            # Edge case: If there's only one or zero updates, just return it.
            return delta_local_updates[0], 0 if return_index else delta_local_updates[0]


        # 1. Flatten each update into one large vector
        keys = list(delta_local_updates[0].keys())
        dim_per_update = sum(delta_local_updates[0][k].numel() for k in keys)
        
        with torch.no_grad():
            updates_flat = torch.empty((num_updates, dim_per_update), device=device)
            for i, update_dict in enumerate(delta_local_updates):
                # Flatten each tensor & concatenate
                flat_list = [update_dict[k].flatten() for k in keys]
                updates_flat[i] = torch.cat(flat_list, dim=0).to(device)

            # 2. Compute pairwise squared distances via:
            #    dist^2(Xi, Xj) = ||Xi||^2 + ||Xj||^2 - 2 Xi·Xj
            #    This yields the full NxN distance matrix in one shot.
            # -----------------------------------------------------
            # Precompute L2 norms (row-wise)
            norms = (updates_flat ** 2).sum(dim=1, keepdim=True)
            # Expand and compute all pairwise squared distances
            distances = norms + norms.T - 2.0 * (updates_flat @ updates_flat.T)
            # Numerical issues may cause tiny negative values; clamp them to 0
            distances.clamp_(min=0)

            # 3. For each row i, sort distances to find the sum of the closest
            #    `non_malicious_count` neighbors (excluding self-distance)
            sorted_distances, _ = distances.sort(dim=1)
            # Exclude self-distance, which should be the 0 at sorted_distances[:, 0]
            # Then sum up the next `non_malicious_count` distances
            scores = sorted_distances[:, 1 : 1 + non_malicious_count].sum(dim=1)

            # 4. Pick the update with the minimum Krum score
            krum_index = scores.argmin().item()


        if return_index:
            return delta_local_updates[krum_index], krum_index
        else:
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

        device = next(iter(delta_local_updates[0].values())).device

        import time
        t = time.time()
        if device == torch.device("cpu"):
            # 1) Move everything to GPU (for speed) - if it fits
            device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for update in delta_local_updates:
                for k, v in update.items():
                    update[k] = v.to(device_gpu)

        # Step 1: Perform iterative Krum selection to create a candidate set
        candidate_set = []
        for _ in range(n - 2 * m):
            kwargs['delta_local_updates'] = delta_local_updates
            candidate, krum_index = self.krum(return_index=True, **kwargs)
            candidate = {k: v.cpu() for k, v in candidate.items()}
            candidate_set.append(candidate)
            delta_local_updates.pop(krum_index)

        with torch.no_grad():
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

        print(time.time() - t)
        return aggregated_weights

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)