import copy
import torch
import wandb
import numpy as np

from server.server_base import BaseServer
from defence.defence import Defence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        lambda_val=(0, 0.05, None),
        c_alpha=1e-4, 
        rho_alpha=0.5, 
        max_line_search_iterations_alpha=0,
        c_beta=1e-2, 
        rho_beta=0.5, 
        max_line_search_iterations_beta=10
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
        lambda_val : tuple
            (start, end, end_epoch). If end_epoch is None => use total_epochs. 
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

        # Build the schedule for lambda_value
        num_steps = lambda_val[-1] if lambda_val[-1] else self.total_epochs
        lambda_range = np.linspace(lambda_val[0], lambda_val[1], num_steps).tolist()
        # If the total epochs exceed num_steps, keep the final value for the remainder
        lambda_range += [lambda_val[1]] * max(0, self.total_epochs - len(lambda_range))

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
                               compute_gradient=True)
            avg_loss_before = np.matmul(np.array(F_T_next).T, np.array(w))

            # Update weights w
            current_lambda = lambda_range[epoch]
            w = self._weight_update(
                G, G_next, F_T_next, w, alpha, beta,
                current_lambda, is_ftotal,
                max_line_search_iterations_beta,
                c_beta,
                rho_beta
            )

            if w.count(0) / len(w) > 0.4:
                beta *= 0.7

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
                "lambda_current": current_lambda,
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
        self, G, G_next, F_T_next, w, alpha, beta, lambda_value, is_ftotal,
        max_line_search_iterations, c_beta, rho_beta, eye_factor=1e-6
    ):
        """
        Updates weight vector w (the distribution across clients) with 
        backtracking line search on beta and projection to a simplex.
        """
        G_flat = self._flatten_tensors(G)
        G_next_flat = self._flatten_tensors(G_next)
        w_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        F_T_next_tensor = torch.tensor(F_T_next, dtype=torch.float32, device=device)

        # G^T G_next plus a tiny regularization on the diagonal
        G_T_G_next = torch.matmul(G_flat.T, G_next_flat)
        G_T_G_next += eye_factor * torch.eye(G_T_G_next.shape[0], device=device)
        G_T_G_next_w = torch.matmul(G_T_G_next, w_tensor)

        if is_ftotal:
            m_next = w_tensor + alpha * beta * G_T_G_next_w - beta * F_T_next_tensor
        else:
            m_next = w_tensor + alpha * beta * G_T_G_next_w

        w_next_normalize = self._sparse_projection_onto_simplex(
            m_next.cpu().numpy(), 
            lambda_value
        )

        # Line search for beta
        beta, w_next_normalize, m_next = self._line_search_for_beta(
            w_tensor, m_next, w_next_normalize, alpha, beta, 
            G_T_G_next_w, F_T_next_tensor, is_ftotal, lambda_value, 
            max_line_search_iterations, c_beta, rho_beta
        )

        self.list_m_next.append(m_next.cpu().numpy())
        self.list_w_next.append(w_next_normalize)
        print(f"beta: {beta}")

        return w_next_normalize

    def _line_search_for_beta(
        self, w_tensor, m_next, w_next_normalize, alpha, beta, 
        G_T_G_next_w, F_T_next_tensor, is_ftotal, lambda_value, 
        max_line_search_iterations, c_beta, rho_beta
    ):
        """
        Armijo line search for beta, used in the weight update step.
        """
        if max_line_search_iterations == 0:
            return beta, w_next_normalize, m_next

        for _ in range(max_line_search_iterations):
            # Evaluate objective
            w_next_tensor = torch.tensor(w_next_normalize, dtype=torch.float32, device=device)
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
                    m_next.cpu().numpy(), 
                    lambda_value
                )
        else:
            print("Line search for beta did not converge within the allotted iterations.")

        return beta, w_next_normalize, m_next

    def _sparse_projection_onto_simplex(self, m_next, lambda_value):
        """
        Projects m_next onto the simplex, ignoring elements with 
        absolute value <= lambda_value.
        """
        # Sort in descending order
        sorted_m = np.sort(m_next)[::-1]
        idxs_desc = np.argsort(m_next)[::-1]

        # Identify elements with magnitude > lambda_value
        valid_mask = np.abs(sorted_m) > lambda_value
        if not np.any(valid_mask):
            return [0.] * len(m_next)

        # Subset to valid elements
        P_L_lambda = sorted_m[valid_mask]
        cumsum_vals = np.cumsum(P_L_lambda)

        # Rho condition
        rhos = (P_L_lambda > (cumsum_vals - 1.0) / np.arange(1, len(P_L_lambda) + 1))
        if np.any(rhos):
            rho_idx = np.where(rhos)[0].max()
            eta = (cumsum_vals[rho_idx] - 1.0) / (rho_idx + 1.0)
        else:
            # fallback if no candidate
            eta = cumsum_vals[-1] / len(P_L_lambda)

        # Final projection
        P_plus = np.maximum(P_L_lambda - eta, 0)
        w_proj = np.zeros_like(m_next)
        w_proj[idxs_desc[valid_mask]] = P_plus

        return w_proj.tolist()
