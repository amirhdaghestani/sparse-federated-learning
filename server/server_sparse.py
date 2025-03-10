import copy
import torch
import wandb
import numpy as np

from server.server_base import BaseServer


class SparseFLServer(BaseServer):
    """
    Implements a Sparse Federated Learning strategy using line searches on alpha and beta,
    projection onto a simplex, and optional partial/total loss weighting.
    """

    def run(
        self, 
        alpha, 
        beta, 
        is_ftotal=True, 
        k_value=(0, None, None, None, None),
        c_alpha=1e-4, 
        rho_alpha=0.5, 
        max_line_search_iterations_alpha=0,
        c_beta=1e-2, 
        rho_beta=0.5, 
        max_line_search_iterations_beta=10,
        estimate_initial_updates=False,
    ):
        """
        Sparse Federated Learning main loop.

        Parameters
        ----------
        alpha : float
            Initial learning rate for model updates.
        beta : float
            Weight update parameter.
        is_ftotal : bool
            Flag to indicate if the entire local loss is used (True) 
            or partial in the weight update step.
        k_value : tuple
            (start_epoch, end_epoch, start_val, end_val). If end_epoch is None => use total_epochs. 
            Controls the threshold used in the simplex projection.
        c_alpha : float
            Line-search parameter for alpha.
        rho_alpha : float
            Reduction factor for alpha.
        max_line_search_iterations_alpha : int
            Maximum iterations for alpha line search.
        c_beta : float
            Line-search parameter for beta.
        rho_beta : float
            Reduction factor for beta.
        max_line_search_iterations_beta : int
            Maximum iterations for beta line search.
        """
        # Initialize weights
        num_clients = len(self.clients)
        w = [1.0 / num_clients] * num_clients

        # Gather initial client gradients and losses
        global_weights = self.global_model.state_dict()
        client_gradients, client_losses = self._gather_client_updates(
            global_weights, 
            epoch=0, 
            lr=alpha, 
            return_avg_loss=True, 
            compute_gradient=True
        )
        G, F_T_next = client_gradients, client_losses

        # For convenience in updates
        G_next = copy.deepcopy(G)

        # Generate k_values list for each epoch
        k_values = [num_clients] * self.total_epochs  # Default to num_clients

        start_decay_epoch = k_value[0]
        end_decay_epoch = k_value[1] if k_value[1] is not None else self.total_epochs
        start_k_value = k_value[2] if k_value[2] is not None else num_clients
        end_k_value = k_value[3] if k_value[3] is not None else int((1 - self.fraction_malicious) * num_clients)

        # Linear decay of k_value from start_k_value to end_k_value over the epochs
        if start_decay_epoch < end_decay_epoch:
            for epoch in range(start_decay_epoch, end_decay_epoch + 1):
                progress = (epoch - start_decay_epoch) / (end_decay_epoch - start_decay_epoch)
                k_values[epoch] = int(start_k_value + progress * (end_k_value - start_k_value))

        # Preserve end_k_value after end_decay_epoch
        for epoch in range(end_decay_epoch + 1, self.total_epochs):
            k_values[epoch] = end_k_value
        
        # Max Bound:
        maximum_weight_bound = None
        if k_value[4] is not None:
            maximum_weight_bound = k_value[4]

        # Main loop
        for epoch in range(self.total_epochs):
            print(f"Epoch {epoch+1}/{self.total_epochs}")
            wandb.log({"epoch": epoch+1})

            # Optionally line-search for alpha
            alpha = self._line_search_alpha(
                alpha, G, F_T_next, w, c_alpha, rho_alpha, epoch,
                max_iteration=max_line_search_iterations_alpha
            )
            print(f"alpha: {alpha}")
            wandb.log({"alpha": alpha})

            # Save a copy of the current global params 
            params_copy = {k: v.clone() for k, v in self.global_model.state_dict().items()}

            # Update global model with G
            self._theta_update(G=G, G_next=G_next, F_T_next=F_T_next, w=w, alpha=alpha, epoch=epoch,
                               compute_gradient=(not(estimate_initial_updates)))
            avg_loss_before = np.matmul(np.array(F_T_next).T, np.array(w))

            # Update weights w
            current_k_value = k_values[epoch]
            w = self._weight_update(
                G, G_next, F_T_next, w, alpha, beta,
                current_k_value, maximum_weight_bound,
                is_ftotal,
                max_line_search_iterations_beta,
                c_beta,
                rho_beta
            )

            # Second model update using new w
            self._theta_update(G=G, G_next=G_next, F_T_next=F_T_next, w=w, alpha=alpha, epoch=epoch,
                               params_copy=params_copy, compute_gradient=True)
            avg_loss_after = np.matmul(np.array(F_T_next).T, np.array(w))
    
            print(f"Average Loss Before Weight Update: {avg_loss_before}")
            print(f"Average Loss After Weight Update: {avg_loss_after}")
            print(f"Sparse Weights: {w}")
            wandb.log({
                "avg_loss_before_weight_update": float(avg_loss_before),
                "avg_loss_after_weight_update": float(avg_loss_after),
                "k_value_current": current_k_value,
                "beta": float(beta)
            })

            # Move to next iteration
            G = copy.deepcopy(G_next)
            # Evaluate periodically
            if epoch % self.evaluate_each_epoch == 0:
                test_acc, test_loss = self.calculate_accuracy(is_fedavg=False)
                wandb.log({"test_accuracy": test_acc, "test_loss": test_loss})

    def _line_search_alpha(self, alpha, G, F_T_next, w, c, rho, epoch, max_iteration=3):
        """
        Armijo line-search for alpha. 
        Decreases alpha by factor rho if improvement is insufficient.
        """
        if max_iteration == 0:
            return alpha  # Bypass line search

        params_new = {k: v.clone() for k, v in self.global_model.state_dict().items()}

        # Flatten aggregated gradient for the check
        for _ in range(max_iteration):
            agg_grad_vector = []
            with torch.no_grad():
                for idx, (name, param) in enumerate(self.global_model.named_parameters()):
                    agg_grad = torch.zeros_like(param)
                    for client_idx, grad_list in enumerate(G):
                        agg_grad += w[client_idx] * grad_list[idx]
                    params_new[name] = param - alpha * agg_grad
                    agg_grad_vector.append(agg_grad)

            # Evaluate the new loss
            new_client_losses = self._gather_client_updates(
                params_new, 
                epoch, 
                lr=alpha, 
                return_avg_loss=True, 
                compute_gradient=False
            )[1]
            L_new, L_old = np.mean(new_client_losses), np.mean(F_T_next)

            agg_grad_tensor = torch.cat([v.view(-1) for v in agg_grad_vector])
            # Armijo condition
            if L_new <= L_old - c * alpha * torch.norm(agg_grad_tensor) ** 2:
                return alpha
            alpha *= rho
        else:
            print("Line search for alpha failed to converge.")
        return alpha

    def _theta_update(self, G, G_next, F_T_next, w, alpha, epoch, params_copy=None, compute_gradient=True):
        """
        Updates self.global_model using aggregated gradients from G (weighted by w),
        then gathers new local updates in G_next, F_T_next.
        """
        with torch.no_grad():
            for idx, (name, param) in enumerate(self.global_model.named_parameters()):
                # Initialize aggregated gradient
                agg_grad = torch.zeros_like(param)

                # Aggregate weighted gradients from all clients
                for client_idx, grad_list in enumerate(G):
                    agg_grad += w[client_idx] * grad_list[idx]

                # Update parameters
                if params_copy is None:
                    # Direct update to the parameter
                    param -= alpha * agg_grad
                else:
                    # Update from a copy of the parameters
                    if name not in params_copy:
                        raise ValueError(f"Parameter {name} not found in params_copy.")
                    param.copy_(params_copy[name] - alpha * agg_grad)

        updated_weights = self.global_model.state_dict()

        # Gather new local updates
        updated_grads, updated_losses = self._gather_client_updates(
            updated_weights,
            epoch=epoch,
            lr=alpha,
            return_avg_loss=True,
            compute_gradient=compute_gradient
        )

        # Overwrite inputs in place
        G_next[:] = updated_grads
        F_T_next[:] = updated_losses

    def _weight_update(
        self, G, G_next, F_T_next, w, alpha, beta, k_value, maximum_weight_bound, is_ftotal,
        max_line_search_iterations, c_beta, rho_beta, eye_factor=1e-6
    ):
        """
        Updates weight vector w (the distribution across clients) with 
        backtracking line search on beta and projection to a simplex.
        """
        G_flat = self._flatten_tensors(G)
        G_next_flat = self._flatten_tensors(G_next)
        w_tensor = torch.tensor(w, dtype=torch.float32, device=self.device)
        F_T_next_tensor = torch.tensor(F_T_next, dtype=torch.float32, device=self.device)

        # G^T G_next plus a tiny regularization on the diagonal
        G_T_G_next = torch.matmul(G_flat.T, G_next_flat)
        G_T_G_next += eye_factor * torch.eye(G_T_G_next.shape[0], device=self.device)
        G_T_G_next_w = torch.matmul(G_T_G_next, w_tensor)

        if is_ftotal:
            m_next = w_tensor + alpha * beta * G_T_G_next_w - beta * F_T_next_tensor
        else:
            m_next = w_tensor + alpha * beta * G_T_G_next_w

        w_next_normalize = self._sparse_projection_onto_simplex(
            m_next=m_next.cpu().numpy(),
            k_value=k_value,
            t=maximum_weight_bound,
        )

        # Line search for beta
        beta, w_next_normalize, m_next = self._line_search_for_beta(
            w_tensor, m_next, w_next_normalize, alpha, beta, 
            G_T_G_next_w, F_T_next_tensor, is_ftotal, k_value, maximum_weight_bound,
            max_line_search_iterations, c_beta, rho_beta
        )

        self.list_m_next.append(m_next.cpu().numpy())
        self.list_w_next.append(w_next_normalize)
        print(f"beta: {beta}")

        return w_next_normalize

    def _line_search_for_beta(
        self, w_tensor, m_next, w_next_normalize, alpha, beta, 
        G_T_G_next_w, F_T_next_tensor, is_ftotal, k_value, maximum_weight_bound,
        max_line_search_iterations, c_beta, rho_beta
    ):
        """
        Armijo line search for beta, used in the weight update step.
        """
        if max_line_search_iterations == 0:
            return beta, w_next_normalize, m_next

        for _ in range(max_line_search_iterations):
            # Evaluate objective
            w_next_tensor = torch.tensor(w_next_normalize, dtype=torch.float32, device=self.device)
            f_new = 0.5 * torch.norm(w_next_tensor - m_next) ** 2
            f_current = 0.5 * torch.norm(w_tensor - m_next) ** 2
            grad_f = torch.abs((w_tensor - m_next).dot(w_next_tensor - w_tensor))

            # Armijo condition
            if f_new <= f_current - c_beta * beta * grad_f:
                break
            else:
                beta *= rho_beta
                if is_ftotal:
                    m_next = w_tensor + alpha * beta * G_T_G_next_w - beta * F_T_next_tensor
                else:
                    m_next = w_tensor + alpha * beta * G_T_G_next_w
                w_next_normalize = self._sparse_projection_onto_simplex(
                    m_next=m_next.cpu().numpy(), 
                    k_value=k_value,
                    t=maximum_weight_bound,
                )
        else:
            print("Line search for beta did not converge within the allotted iterations.")

        return beta, w_next_normalize, m_next

    def _sparse_projection_onto_simplex(self, *args, **kwargs):
        """
        Alias for the simplex projection method.
        """
        t_val = kwargs.get("t", None)
        if t_val is not None:
            return self._sparse_projection_capped_simplex(*args, **kwargs)
        else:
            kwargs.pop("t", None)
            return self._sparse_projection_onto_unit_simplex(*args, **kwargs)

    def _sparse_projection_onto_unit_simplex(self, m_next, k_value):
        """
        Projects m_next onto the simplex by keeping the largest k_value elements.
        """
        # Make sure m_next is a NumPy array.
        m_next = np.array(m_next, dtype=float)

        if k_value < 1 or k_value > len(m_next):
            return [0.] * len(m_next)

        # Sort in descending order
        idxs_desc = np.argsort(m_next)[::-1]
        sorted_m = m_next[idxs_desc]  # Now valid because m_next is a NumPy array

        # Select the top k elements
        top_k_idxs = idxs_desc[:k_value]
        P_L_lambda = sorted_m[:k_value]

        # Compute cumulative sum
        cumsum_vals = np.cumsum(P_L_lambda)

        # Find rho index
        rhos = (P_L_lambda > (cumsum_vals - 1.0) / np.arange(1, k_value + 1))
        if np.any(rhos):
            rho_idx = np.where(rhos)[0].max()
            eta = (cumsum_vals[rho_idx] - 1.0) / (rho_idx + 1.0)
        else:
            eta = cumsum_vals[-1] / k_value  # fallback

        # Apply projection
        P_plus = np.maximum(P_L_lambda - eta, 0)

        # Create projected output
        w_proj = np.zeros_like(m_next)
        w_proj[top_k_idxs] = P_plus

        return w_proj.tolist()
    
    def _sparse_projection_capped_simplex(self, m_next, k_value, t):
        """
        Projects m_next onto the capped simplex:

            min_w  0.5 * ||w - m_next||^2
            s.t.   sum_i w_i = 1,   0 <= w_i <= t

        Optionally, only the top 'k_value' largest components
        of m_next can be non-zero; the rest are forced to 0.

        Parameters
        ----------
        m_next : array-like of shape (n,)
            Input vector.
        t : float
            Upper bound (cap) for each coordinate w_i.
        k_value : int
            Number of largest components to consider for the projection.
            All others become 0.

        Returns
        -------
        w_proj : list of float
            A projected vector of the same length as m_next, satisfying
            sum(w_proj) = 1 (if feasible) and each w_proj[i] <= t.
        """
        m_next = np.array(m_next, dtype=float)
        n_full = len(m_next)
        k = 1.0

        if k_value < 1:
            raise ValueError("k_value < 1 is not valid when sum must be 1.")

        idxs_desc = np.argsort(m_next)[::-1]
        k_value = min(k_value, n_full)
        valid_mask = idxs_desc[:k_value]
        y0 = m_next[valid_mask]

        if k_value * t < k:
            raise ValueError("The sum=1 constraint is infeasible with k_value * t < 1.")

        y0_scaled = y0 / t
        k_scaled = k / t
        x_part = np.zeros(k_value, dtype=float)
        idx_asc = np.argsort(y0_scaled)
        y_asc = y0_scaled[idx_asc]
        s_cum = np.cumsum(y_asc)
        y_asc = np.append(y_asc, np.inf)

        for b in range(1, k_value + 1):
            gamma = (k_scaled + b - k_value - s_cum[b - 1]) / b
            if (y_asc[0] + gamma > 0) and (y_asc[b - 1] + gamma < 1) and (y_asc[b] + gamma >= 1):
                xtmp = np.concatenate((y_asc[:b] + gamma, np.ones(k_value - b)))
                xtmp *= t
                x_part[idx_asc] = xtmp
                w_proj = np.zeros(n_full, dtype=float)
                w_proj[valid_mask] = x_part
                if np.isclose(np.sum(w_proj), 1.0, atol=1e-7):
                    return w_proj.tolist()

        for a in range(1, k_value):
            for b in range(a + 1, k_value + 1):
                gamma = (k_scaled + b - k_value + s_cum[a - 1] - s_cum[b - 1]) / (b - a)
                if (y_asc[a - 1] + gamma <= 0) and (y_asc[a] + gamma > 0) and (y_asc[b - 1] + gamma < 1) and (y_asc[b] + gamma >= 1):
                    xtmp = np.concatenate((np.zeros(a), y_asc[a:b] + gamma, np.ones(k_value - b)))
                    xtmp *= t
                    x_part[idx_asc] = xtmp
                    w_proj = np.zeros(n_full, dtype=float)
                    w_proj[valid_mask] = x_part
                    if np.isclose(np.sum(w_proj), 1.0, atol=1e-7):
                        return w_proj.tolist()

        needed = 1.0
        x_part = np.zeros(k_value, dtype=float)
        for i in range(k_value):
            if needed >= t:
                x_part[i] = t
                needed -= t
            else:
                x_part[i] = needed
                needed = 0.0
                break

        w_proj = np.zeros(n_full, dtype=float)
        w_proj[valid_mask] = x_part

        if not np.isclose(np.sum(w_proj), 1.0, atol=1e-7):
            raise ValueError("Projection failed: sum of w_proj is not close to 1.0")

        return w_proj.tolist()
