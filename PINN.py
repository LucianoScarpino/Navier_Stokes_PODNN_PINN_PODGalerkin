import numpy as np
import torch

import torch.nn as nn
import pickle
import os

from pypolydim import polydim
from other_utilities import plot_FOM_solution


class ResidualFNNBlock(nn.Module):
    def __init__(self, width, activation=nn.Tanh):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            activation(),
            nn.Linear(width, width)
        )
        self.activation = activation()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ParametricPINNNetwork(nn.Module):
    def __init__(
        self,
        input_dim=4,
        output_dim=3,
        width=128,
        latent_dim=64,
        n_residual_blocks=4,
        activation=nn.Tanh
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, width),
            activation(),
            nn.Linear(width, latent_dim),
            activation()
        )

        self.latent_lift = nn.Sequential(
            nn.Linear(latent_dim, width),
            activation()
        )

        self.residual_core = nn.Sequential(
            *[ResidualFNNBlock(width, activation=activation) for _ in range(n_residual_blocks)]
        )

        self.decoder = nn.Sequential(
            nn.Linear(width, width),
            activation(),
            nn.Linear(width, output_dim)
        )

    def forward(self, input_points):
        latent = self.encoder(input_points)
        features = self.latent_lift(latent)
        features = self.residual_core(features)
        output = self.decoder(features)
        return output


class PINN_Methods(object):
    def __init__(self, mu0_range, mu1_range, fom_data, device=None):
        self.mu0_range = mu0_range
        self.mu1_range = mu1_range
        self.fom_data = fom_data

        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        elif device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "mps":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def solve_PINN(self, model, x, y, mu0, mu1, plot_solution=False):
        """
        Evaluate the trained PINN at prescribed spatial points and parameters.

        Parameters
        ----------
        model : torch.nn.Module
            Trained PINN model.
        x, y : array-like
            Spatial coordinates where the solution is evaluated.
        mu0, mu1 : float
            Parameter values.

        Returns
        -------
        dict
            Dictionary containing u_x, u_y, p and the global stacked solution.
        """
        model = model.to(self.device)
        model.eval()

        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device).reshape(-1, 1)
        y_tensor = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)

        mu0_tensor = torch.full_like(x_tensor, float(mu0))
        mu1_tensor = torch.full_like(x_tensor, float(mu1))

        evaluation_points = torch.cat(
            [x_tensor, y_tensor, mu0_tensor, mu1_tensor],
            dim=1
        )

        with torch.no_grad():
            network_output = model(evaluation_points)

        u_x = network_output[:, 0].detach().cpu().numpy()
        u_y = network_output[:, 1].detach().cpu().numpy()
        p = network_output[:, 2].detach().cpu().numpy()

        global_solution = np.concatenate([u_x, u_y, p])

        if plot_solution:
            self.plot_PINN_solution(
                model=model,
                mu0=mu0,
                mu1=mu1,
                vtk_utilities=self.fom_data["vtk_utilities"]
            )

        return {
            "u": global_solution,
            "u_x": u_x,
            "u_y": u_y,
            "p": p,
            "converged": np.nan,
            "iterations": np.nan,
            "relative_increment": np.nan
        }

    def solve_PINN_on_FEM_dofs(self, model, mu0, mu1, plot_solution=False):
        """
        Evaluate the trained PINN on the same FEM DOFs used by the FOM.

        The returned global vector has the same structure used by FOM, POD-Galerkin
        and PODNN:
            [u_x_on_speed_dofs, u_y_on_speed_dofs, p_on_pressure_dofs]

        This method is intended for FOM-vs-PINN metric evaluation.
        """
        if self.fom_data is None:
            raise ValueError("solve_PINN_on_FEM_dofs requires self.fom_data.")

        model = model.to(self.device)
        model.eval()

        def evaluate_component(x, y, component_index):
            point = torch.tensor(
                [[float(x), float(y), float(mu0), float(mu1)]],
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                value = model(point)[0, component_index]

            return float(value.detach().cpu().numpy())

        def u_x_function(x, y, z):
            return evaluate_component(x, y, 0)

        def u_y_function(x, y, z):
            return evaluate_component(x, y, 1)

        def p_function(x, y, z):
            return evaluate_component(x, y, 2)

        u_x_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            u_x_function
        ).function_dofs

        u_y_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            u_y_function
        ).function_dofs

        p_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["pressure_dofs_data"],
            self.fom_data["pressure_reference_element_data"],
            p_function
        ).function_dofs

        global_solution = np.concatenate([u_x_numeric, u_y_numeric, p_numeric])

        if plot_solution:
            self.plot_PINN_solution(
                model=model,
                mu0=mu0,
                mu1=mu1,
                vtk_utilities=self.fom_data["vtk_utilities"],
            )

        return {
            "u": global_solution,
            "u_x": u_x_numeric,
            "u_y": u_y_numeric,
            "p": p_numeric,
            "converged": np.nan,
            "iterations": np.nan,
            "relative_increment": np.nan
        }

    def plot_PINN_solution(
        self,
        model,
        mu0,
        mu1,
        vtk_utilities,
        export_solution_path="./Export/Solution/PINN",
        plot_path="./Plots/PINN"
        ):
        """
        Plot/export a reconstructed PINN solution on the FEM mesh.

        This method uses self.fom_data, exactly as the POD-Galerkin and PODNN
        plotting utilities do. Therefore it is available only when the PINN object
        has been created inside the offline/main workflow with fom_data available.
        """
        if self.fom_data is None:
            raise ValueError(
                "plot_PINN_solution requires self.fom_data. It cannot be used with a pure PINN model loaded from pickle."
            )

        if not os.path.exists(export_solution_path):
            os.makedirs(export_solution_path)

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        model = model.to(self.device)
        model.eval()

        def evaluate_component(x, y, component_index):
            point = torch.tensor(
                [[float(x), float(y), float(mu0), float(mu1)]],
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                value = model(point)[0, component_index]

            return float(value.detach().cpu().numpy())

        def u_x_function(x, y, z):
            return evaluate_component(x, y, 0)

        def u_y_function(x, y, z):
            return evaluate_component(x, y, 1)

        def p_function(x, y, z):
            return evaluate_component(x, y, 2)

        u_x_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            u_x_function
        ).function_dofs

        u_y_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            u_y_function
        ).function_dofs

        p_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["pressure_dofs_data"],
            self.fom_data["pressure_reference_element_data"],
            p_function
        ).function_dofs

        plot_FOM_solution(
            mesh=self.fom_data["mesh"],
            speed_dofs_data=self.fom_data["speed_dofs_data"],
            u_x_numeric=u_x_numeric,
            u_x_strong=self.fom_data["u_x_strong"],
            u_y_numeric=u_y_numeric,
            u_y_strong=self.fom_data["u_y_strong"],
            pressure_dofs_data=self.fom_data["pressure_dofs_data"],
            p_numeric=p_numeric,
            p_strong=self.fom_data["p_strong"],
            vtk_utilities=vtk_utilities,
            export_solution_path=export_solution_path,
            plot_path=plot_path,
            method='PINN'
        )

    def build_network(
        self,
        width=128,
        latent_dim=64,
        n_residual_blocks=4,
        activation=nn.Tanh
    ):
        """
        Build the neural architecture used by the PINN.

        Input:
            (x, y, mu0, mu1)

        Output:
            (u_x, u_y, p)

        The architecture is an encoder-core-decoder FNN:
            input -> latent representation -> residual feature core -> physical output.

        This is not an autoencoder in the classical unsupervised sense, because the
        input is not reconstructed. However, it uses an encoder/latent/decoder structure
        to learn a compact internal representation of the parametric solution map.
        """
        model = ParametricPINNNetwork(
            input_dim=4,
            output_dim=3,
            width=width,
            latent_dim=latent_dim,
            n_residual_blocks=n_residual_blocks,
            activation=activation
        ).to(self.device)

        return model

    def sample_interior_points(self, n_points):
        """
        Sample collocation points inside the parametric space-time-independent domain.

        Each sampled point has the form:
            (x, y, mu0, mu1)

        with:
            (x, y) in Omega = (0, 1)^2
            mu0 in self.mu0_range
            mu1 in self.mu1_range
        """
        x = torch.rand((n_points, 1), device=self.device)
        y = torch.rand((n_points, 1), device=self.device)

        mu0_min, mu0_max = self.mu0_range
        mu1_min, mu1_max = self.mu1_range

        mu0 = mu0_min + (mu0_max - mu0_min) * torch.rand((n_points, 1), device=self.device)
        mu1 = mu1_min + (mu1_max - mu1_min) * torch.rand((n_points, 1), device=self.device)

        interior_points = torch.cat([x, y, mu0, mu1], dim=1)
        interior_points.requires_grad_(True)

        return interior_points

    def sample_boundary_points(self, n_points):
        """
        Sample collocation points on the boundary of Omega = (0, 1)^2.

        Each sampled point has the form:
            (x, y, mu0, mu1)

        The boundary is split uniformly among the four sides:
            x = 0, x = 1, y = 0, y = 1.

        These points are used to impose the no-slip Dirichlet condition:
            u_x = 0, u_y = 0 on boundary(Omega).
        """
        n_per_side = n_points // 4
        remainder = n_points - 4 * n_per_side

        counts = [n_per_side, n_per_side, n_per_side, n_per_side]
        for i in range(remainder):
            counts[i] += 1

        mu0_min, mu0_max = self.mu0_range
        mu1_min, mu1_max = self.mu1_range

        boundary_points = []

        # x = 0
        y = torch.rand((counts[0], 1), device=self.device)
        x = torch.zeros_like(y)
        mu0 = mu0_min + (mu0_max - mu0_min) * torch.rand((counts[0], 1), device=self.device)
        mu1 = mu1_min + (mu1_max - mu1_min) * torch.rand((counts[0], 1), device=self.device)
        boundary_points.append(torch.cat([x, y, mu0, mu1], dim=1))

        # x = 1
        y = torch.rand((counts[1], 1), device=self.device)
        x = torch.ones_like(y)
        mu0 = mu0_min + (mu0_max - mu0_min) * torch.rand((counts[1], 1), device=self.device)
        mu1 = mu1_min + (mu1_max - mu1_min) * torch.rand((counts[1], 1), device=self.device)
        boundary_points.append(torch.cat([x, y, mu0, mu1], dim=1))

        # y = 0
        x = torch.rand((counts[2], 1), device=self.device)
        y = torch.zeros_like(x)
        mu0 = mu0_min + (mu0_max - mu0_min) * torch.rand((counts[2], 1), device=self.device)
        mu1 = mu1_min + (mu1_max - mu1_min) * torch.rand((counts[2], 1), device=self.device)
        boundary_points.append(torch.cat([x, y, mu0, mu1], dim=1))

        # y = 1
        x = torch.rand((counts[3], 1), device=self.device)
        y = torch.ones_like(x)
        mu0 = mu0_min + (mu0_max - mu0_min) * torch.rand((counts[3], 1), device=self.device)
        mu1 = mu1_min + (mu1_max - mu1_min) * torch.rand((counts[3], 1), device=self.device)
        boundary_points.append(torch.cat([x, y, mu0, mu1], dim=1))

        boundary_points = torch.cat(boundary_points, dim=0)
        boundary_points.requires_grad_(True)

        return boundary_points

    def source_term(self, points):
        """
        Compute the parametric forcing term f(x, y; mu1) in PyTorch.

        Input:
            points[:, 0] = x
            points[:, 1] = y
            points[:, 2] = mu0
            points[:, 3] = mu1

        Output:
            f_x, f_y

        The returned tensors are used in the momentum residuals of the PINN.
        """
        x = points[:, 0:1]
        y = points[:, 1:2]
        mu1 = points[:, 3:4]

        pi = torch.pi

        f_x = -(
            mu1**3 * pi**2 * torch.cos(mu1**2 * pi * x)
            - mu1**2 * pi**2
        ) * torch.sin(mu1 * pi * y) * torch.cos(mu1 * pi * y) \
        + (
            mu1 * pi * torch.cos(mu1 * pi * x) * torch.cos(mu1 * pi * y)
        )

        f_y = -(
            -mu1**3 * pi**2 * torch.cos(mu1**2 * pi * y)
            + mu1**2 * pi**2
        ) * torch.sin(mu1 * pi * x) * torch.cos(mu1 * pi * x) \
        + (
            -mu1 * pi * torch.sin(mu1 * pi * x) * torch.sin(mu1 * pi * y)
        )

        return f_x, f_y

    def gradient(self, output, input_points):
        """
        Compute the gradient of a scalar network output with respect to the input points.

        Parameters
        ----------
        output : torch.Tensor
            Tensor with shape (n_points, 1), for example u_x, u_y, or p.
        input_points : torch.Tensor
            Tensor with shape (n_points, 4), containing (x, y, mu0, mu1).

        Returns
        -------
        torch.Tensor
            Tensor with shape (n_points, 4). Columns correspond to derivatives with
            respect to (x, y, mu0, mu1).
        """
        return torch.autograd.grad(
            output,
            input_points,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]

    def compute_pde_residuals(self, model, interior_points):
        """
        Compute the Navier-Stokes residuals at interior collocation points.

        The network approximates:
            (x, y, mu0, mu1) -> (u_x, u_y, p)

        The residuals are:
            r_x   = -mu0 * Delta(u_x) + u_x * d_x(u_x) + u_y * d_y(u_x) + d_x(p) - f_x
            r_y   = -mu0 * Delta(u_y) + u_x * d_x(u_y) + u_y * d_y(u_y) + d_y(p) - f_y
            r_div = d_x(u_x) + d_y(u_y)
        """
        network_output = model(interior_points)

        u_x = network_output[:, 0:1]
        u_y = network_output[:, 1:2]
        p = network_output[:, 2:3]

        mu0 = interior_points[:, 2:3]

        grad_u_x = self.gradient(u_x, interior_points)
        grad_u_y = self.gradient(u_y, interior_points)
        grad_p = self.gradient(p, interior_points)

        u_x_x = grad_u_x[:, 0:1]
        u_x_y = grad_u_x[:, 1:2]
        u_y_x = grad_u_y[:, 0:1]
        u_y_y = grad_u_y[:, 1:2]

        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]

        grad_u_x_x = self.gradient(u_x_x, interior_points)
        grad_u_x_y = self.gradient(u_x_y, interior_points)
        grad_u_y_x = self.gradient(u_y_x, interior_points)
        grad_u_y_y = self.gradient(u_y_y, interior_points)

        u_x_xx = grad_u_x_x[:, 0:1]
        u_x_yy = grad_u_x_y[:, 1:2]
        u_y_xx = grad_u_y_x[:, 0:1]
        u_y_yy = grad_u_y_y[:, 1:2]

        laplacian_u_x = u_x_xx + u_x_yy
        laplacian_u_y = u_y_xx + u_y_yy

        f_x, f_y = self.source_term(interior_points)

        residual_x = -mu0 * laplacian_u_x + u_x * u_x_x + u_y * u_x_y + p_x - f_x
        residual_y = -mu0 * laplacian_u_y + u_x * u_y_x + u_y * u_y_y + p_y - f_y
        residual_divergence = u_x_x + u_y_y

        return residual_x, residual_y, residual_divergence


    def compute_loss(
        self,
        model,
        interior_points,
        boundary_points,
        lambda_pde=1.0,
        lambda_divergence=1.0,
        lambda_boundary=10.0,
        lambda_pressure_anchor=1.0
    ):
        """
        Compute the full PINN loss.

        The loss contains four terms:
            1. momentum residual loss inside the domain,
            2. divergence-free loss inside the domain,
            3. no-slip boundary loss on the boundary,
            4. pressure anchor loss p(0,0,mu0,mu1)=0.

        The pressure anchor is needed because pressure is determined up to an additive constant.
        """
        residual_x, residual_y, residual_divergence = self.compute_pde_residuals(
            model,
            interior_points
        )

        zero_residual_x = torch.zeros_like(residual_x)
        zero_residual_y = torch.zeros_like(residual_y)
        zero_residual_divergence = torch.zeros_like(residual_divergence)

        mse_loss = nn.MSELoss()

        loss_pde = (
            mse_loss(residual_x, zero_residual_x)
            + mse_loss(residual_y, zero_residual_y)
        )

        loss_divergence = mse_loss(residual_divergence, zero_residual_divergence)

        boundary_output = model(boundary_points)
        boundary_u_x = boundary_output[:, 0:1]
        boundary_u_y = boundary_output[:, 1:2]

        loss_boundary = (
            mse_loss(boundary_u_x, torch.zeros_like(boundary_u_x))
            + mse_loss(boundary_u_y, torch.zeros_like(boundary_u_y))
        )

        mu0_anchor = interior_points[:, 2:3]
        mu1_anchor = interior_points[:, 3:4]
        pressure_anchor_points = torch.cat(
            [
                torch.zeros_like(mu0_anchor),
                torch.zeros_like(mu0_anchor),
                mu0_anchor,
                mu1_anchor
            ],
            dim=1
        )

        pressure_anchor_output = model(pressure_anchor_points)
        pressure_anchor = pressure_anchor_output[:, 2:3]
        loss_pressure_anchor = mse_loss(
            pressure_anchor,
            torch.zeros_like(pressure_anchor)
        )

        total_loss = (
            lambda_pde * loss_pde
            + lambda_divergence * loss_divergence
            + lambda_boundary * loss_boundary
            + lambda_pressure_anchor * loss_pressure_anchor
        )

        loss_terms = {
            "total_loss": total_loss,
            "loss_pde": loss_pde,
            "loss_divergence": loss_divergence,
            "loss_boundary": loss_boundary,
            "loss_pressure_anchor": loss_pressure_anchor
        }

        return total_loss, loss_terms


    def train(
        self,
        model,
        n_epochs=5000,
        n_interior=4096,
        n_boundary=1024,
        learning_rate=1.0e-3,
        weight_decay=1.0e-8,
        lambda_pde=1.0,
        lambda_divergence=1.0,
        lambda_boundary=10.0,
        lambda_pressure_anchor=1.0,
        print_every=100
    ):
        """
        Train the PINN by minimizing the physics-informed loss.

        At each epoch new interior and boundary collocation points are sampled.
        The training does not use FOM snapshots: it only uses the PDE residual,
        the no-slip boundary condition, the divergence-free constraint and the
        pressure anchor.
        """
        model = model.to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        history = {
            "total_loss": [],
            "loss_pde": [],
            "loss_divergence": [],
            "loss_boundary": [],
            "loss_pressure_anchor": []
        }

        model.train()

        for epoch in range(1, n_epochs + 1):
            interior_points = self.sample_interior_points(n_interior)
            boundary_points = self.sample_boundary_points(n_boundary)

            optimizer.zero_grad()

            total_loss, loss_terms = self.compute_loss(
                model=model,
                interior_points=interior_points,
                boundary_points=boundary_points,
                lambda_pde=lambda_pde,
                lambda_divergence=lambda_divergence,
                lambda_boundary=lambda_boundary,
                lambda_pressure_anchor=lambda_pressure_anchor
            )

            total_loss.backward()
            optimizer.step()

            history["total_loss"].append(loss_terms["total_loss"].item())
            history["loss_pde"].append(loss_terms["loss_pde"].item())
            history["loss_divergence"].append(loss_terms["loss_divergence"].item())
            history["loss_boundary"].append(loss_terms["loss_boundary"].item())
            history["loss_pressure_anchor"].append(loss_terms["loss_pressure_anchor"].item())

            if print_every is not None and (epoch == 1 or epoch % print_every == 0 or epoch == n_epochs):
                print(
                    f"[PINN] epoch {epoch:05d}/{n_epochs} | "
                    f"total={loss_terms['total_loss'].item():.3e} | "
                    f"pde={loss_terms['loss_pde'].item():.3e} | "
                    f"div={loss_terms['loss_divergence'].item():.3e} | "
                    f"bc={loss_terms['loss_boundary'].item():.3e} | "
                    f"p0={loss_terms['loss_pressure_anchor'].item():.3e}"
                )

        return model, history


    def save_PINN_model(self, model, export_path, metadata=None):
        """
        Save the trained PINN model.

        Only serializable data are saved: network weights, parameter ranges,
        device-independent architecture metadata and optional metadata.
        """
        model_data = {
            "model_state_dict": model.state_dict(),
            "mu0_range": self.mu0_range,
            "mu1_range": self.mu1_range,
            "metadata": metadata if metadata is not None else {}
        }

        export_folder = os.path.dirname(export_path)
        if export_folder != "" and not os.path.exists(export_folder):
            os.makedirs(export_folder)

        with open(export_path, "wb") as file:
            pickle.dump(model_data, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"[PINN] model saved to: {export_path}")

    @staticmethod
    def load_PINN_model(export_path):
        """
        Load a previously saved PINN model dictionary.
        """
        if not os.path.exists(export_path):
            raise FileNotFoundError(
                f"PINN model file not found: {export_path}. Run the offline training first."
            )

        if os.path.getsize(export_path) == 0:
            raise ValueError(
                f"PINN model file is empty: {export_path}. Delete it and regenerate it."
            )

        with open(export_path, "rb") as file:
            model_data = pickle.load(file)

        print(f"[PINN] model loaded from: {export_path}")

        return model_data
