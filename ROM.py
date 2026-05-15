import numpy as np
import scipy
import pickle
import os

from pypolydim import polydim
from other_utilities import make_np_sparse


class ROM_Methods(object):
    def __init__(self,fom_sol,operators,fom_data,training_set):
        self.training_set = training_set
        self.operators = operators
        self.fom_sol = fom_sol
        self.fom_data = fom_data

    def save_reduced_model(self, reduced_data, export_path, metadata=None):
        reduced_model = {
            "reduced_data": reduced_data,
            "fom_data": self.fom_data,
            "metadata": metadata if metadata is not None else {}
        }

        temporary_path = export_path + ".tmp"

        with open(temporary_path, "wb") as file:
            pickle.dump(reduced_model, file, protocol=pickle.HIGHEST_PROTOCOL)

        import os
        os.replace(temporary_path, export_path)

        print(f"[ROM] Reduced model saved to: {export_path}")

    @staticmethod
    def load_reduced_model(export_path):

        if not os.path.exists(export_path):
            raise FileNotFoundError(
                f"Reduced model file not found: {export_path}. Run the offline script first."
            )

        if os.path.getsize(export_path) == 0:
            raise ValueError(
                f"Reduced model file is empty: {export_path}. Delete it and rerun the offline script to regenerate it."
            )

        with open(export_path, "rb") as file:
            reduced_model = pickle.load(file)

        print(f"[ROM] Reduced model loaded from: {export_path}")

        return reduced_model

    def solve_POD_Galerkin(self, reduced_data, mu0, mu1, newton_tol=1e-6, max_iterations=10):
        """
        Online POD-Galerkin solver.

        This method intentionally works only with reduced quantities.
        The nonlinear reduced convection operators must be precomputed offline
        and stored in reduced_data before calling this method.
        """

        V = reduced_data["V"]
        J_A_N = reduced_data["J_A_N"]
        J_B_N = reduced_data["J_B_N"]
        C_jacobian_tensor = reduced_data["C_jacobian_tensor"]
        C_residual_tensor = reduced_data["C_residual_tensor"]

        assembler = self.fom_data["assembler"]
        f_N = self.assemble_reduced_rhs(assembler,mu1,V)

        n_red = V.shape[1]
        a_k = np.zeros(n_red)

        increment_norm = 1.0
        solution_norm = 1.0
        num_iteration = 1

        J_S_N = mu0 * J_A_N - J_B_N

        while num_iteration < max_iterations and increment_norm > newton_tol * solution_norm:
            C_residual_N = np.einsum("ijk,j,k->i", C_residual_tensor, a_k, a_k)
            J_C_N = np.einsum("ijk,k->ij", C_jacobian_tensor, a_k)

            rhs_N = f_N - C_residual_N - J_S_N @ a_k
            da = np.linalg.solve(J_S_N + J_C_N, rhs_N)

            a_k = a_k + da

            increment_norm = np.linalg.norm(da)
            solution_norm = max(np.linalg.norm(a_k), 1e-14)

            print(
                f"[ROM Newton] iter {num_iteration:02d}/{max_iterations} | "
                f"||da||/||a||={increment_norm / solution_norm:.3e} | "
                f"tol={newton_tol:.1e}"
            )

            num_iteration += 1

        U_N = V @ a_k


        print("\n" + "-" * 100)
        print("[Online ROM] Solving POD-Galerkin reduced problem")
        print(f"[Online ROM] mu0={mu0}, mu1={mu1}, newton_tol={newton_tol}, max_iterations={max_iterations}")
        print("\n[Online ROM] Completed")
        print(f"[Online ROM] converged={increment_norm <= newton_tol * solution_norm}")
        print(f"[Online ROM] iterations={num_iteration - 1}")
        print(f"[Online ROM] relative_increment={increment_norm / solution_norm}")
        print(f"[Online ROM] reduced_coefficients_shape={a_k.shape}")
        print(f"[Online ROM] reconstructed_solution_shape={U_N.shape}")
        print("=" * 100)

        return {
            "u": U_N,
            "coefficients": a_k,
            "iterations": num_iteration - 1,
            "relative_increment": increment_norm / solution_norm,
            "converged": increment_norm <= newton_tol * solution_norm
        }
    
    def reduce(self,
                solver,
                p_boundary_info,
                u_boundary_info,
                tol=1e-6,
                N_max=20
                ):
        
        A_op = self.operators['A_operator']
        B_x_op = self.operators["B_x_operator"]
        B_y_op = self.operators["B_y_operator"]
        J_A = self.operators["J_A"]
        J_B = self.operators["J_B"]

        speed_n_dofs = self.fom_sol['speed_n_dofs']
        pressure_n_dofs = self.fom_sol['pressure_n_dofs']

        # Inner product = X1 + X2 (Grad-Grad)
        inner_product_u,B_1,B_2 = self.assemble_supremizer_matricies(A_op,
                                                                     B_x_op,
                                                                     B_y_op,
                                                                     speed_n_dofs,
                                                                     pressure_n_dofs)
        snapshot_matrix_u, snapshot_matrix_s,\
              snapshot_matrix_p, C_u, C_s, C_p = self.create_POD_matricies( solver,
                                                                            p_boundary_info,
                                                                            u_boundary_info,
                                                                            speed_n_dofs,
                                                                            inner_product_u,
                                                                            B_1,
                                                                            B_2,
                                                                            tol=1e-6,
                                                                            N_max=20
                                                                            )
        
        N_u, eigs_u = self.eig_analysis(C_u, N_max=N_max, tol=tol)
        N_s, eigs_s = self.eig_analysis(C_s, N_max=N_max, tol=tol)
        N_p, eigs_p = self.eig_analysis(C_p, N_max=N_max, tol=tol)

        basis_functions_u = self.create_basis_functions_matrix(N_u, snapshot_matrix_u, eigs_u, inner_product=inner_product_u)
        basis_functions_s = self.create_basis_functions_matrix(N_s, snapshot_matrix_s, eigs_s, inner_product=inner_product_u)
        basis_functions_p = self.create_basis_functions_matrix(N_p, snapshot_matrix_p, eigs_p)

        global_basis_function_matrix = np.zeros((basis_functions_u.shape[0] + basis_functions_p.shape[0],N_u + N_s + N_p))
        global_basis_function_matrix[0:basis_functions_u.shape[0], 0:N_u] = basis_functions_u
        global_basis_function_matrix[0:basis_functions_u.shape[0], N_u : N_u + N_s] = basis_functions_s
        global_basis_function_matrix[basis_functions_u.shape[0]:, N_u + N_s:] = basis_functions_p

        global_basis_function_matrix_no_sup = np.zeros((basis_functions_u.shape[0] + basis_functions_p.shape[0],N_u + N_p))
        global_basis_function_matrix_no_sup[0:basis_functions_u.shape[0], 0:N_u] = basis_functions_u
        global_basis_function_matrix_no_sup[basis_functions_u.shape[0]:, N_u:] = basis_functions_p

        V = global_basis_function_matrix
        
        J_A_N = self.assemble_reduced_matrix(V, J_A)
        J_B_N = self.assemble_reduced_matrix(V, J_B)

        C_residual_tensor, C_jacobian_tensor = self.precompute_reduced_convective_tensors(solver,V)

        print("\n" + "-" * 100)
        print("[Offline ROM] Starting POD basis construction and reduced operator assembly")
        print("-" * 100)

        return {
            "V": V,
            "J_A_N": J_A_N,
            "J_B_N": J_B_N,
            "snapshot_matrix_u": snapshot_matrix_u,
            "snapshot_matrix_s": snapshot_matrix_s,
            "snapshot_matrix_p": snapshot_matrix_p,
            "N_u": N_u,
            "N_s": N_s,
            "N_p": N_p,
            "basis_functions_u": basis_functions_u,
            "basis_functions_s": basis_functions_s,
            "basis_functions_p": basis_functions_p,
            "C_jacobian_tensor": C_jacobian_tensor,
            "C_residual_tensor": C_residual_tensor
            }

    def precompute_reduced_convective_tensors(self, solver, V):
        """
        Offline precomputation of the reduced quadratic convection term.

        This method uses the FEM data already generated by the FOM solver and
        stored in self.fom_data.

        The residual tensor satisfies
            C_N(a)_i = sum_{j,k} C_residual_tensor[i,j,k] a_j a_k.

        The Jacobian tensor satisfies
            J_C_N(a)_{i,j} = sum_k C_jacobian_tensor[i,j,k] a_k.
        """

        geometry_utilities = self.fom_data["geometry_utilities"]
        mesh = self.fom_data["mesh"]
        mesh_geometric_data = self.fom_data["mesh_geometric_data"]
        speed_dofs_data = self.fom_data["speed_dofs_data"]
        speed_reference_element_data = self.fom_data["speed_reference_element_data"]
        u_x_strong = self.fom_data["u_x_strong"]
        u_y_strong = self.fom_data["u_y_strong"]

        speed_n_dofs = self.fom_sol["speed_n_dofs"]
        pressure_n_dofs = self.fom_sol["pressure_n_dofs"]

        n_red = V.shape[1]
        C_residual_tensor = np.zeros((n_red, n_red, n_red))

        def project_convective_rhs(full_vector):
            u_x_numeric = full_vector[0:speed_n_dofs]
            u_y_numeric = full_vector[speed_n_dofs:2 * speed_n_dofs]

            c_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_ns_operators(
                geometry_utilities,
                mesh,
                mesh_geometric_data,
                speed_dofs_data,
                speed_reference_element_data,
                u_x_numeric,
                u_y_numeric,
                u_x_strong,
                u_y_strong
            )

            f_C = np.concatenate([
                c_operator.convective_rhs,
                np.zeros(pressure_n_dofs)
            ])

            return self.assemble_reduced_vector(V, f_C)

        print("Precomputing reduced convective tensors...")

        single_terms = []
        for j in range(n_red):
            single_terms.append(project_convective_rhs(V[:, j]))

        for j in range(n_red):
            C_residual_tensor[:, j, j] = single_terms[j]

            for k in range(j + 1, n_red):
                mixed_term = 0.5 * (
                    project_convective_rhs(V[:, j] + V[:, k])
                    - single_terms[j]
                    - single_terms[k]
                )

                C_residual_tensor[:, j, k] = mixed_term
                C_residual_tensor[:, k, j] = mixed_term

        C_jacobian_tensor = C_residual_tensor + np.swapaxes(C_residual_tensor, 1, 2)

        return C_residual_tensor, C_jacobian_tensor

    def assemble_reduced_rhs(self, assembler, mu1, V):

        f_x_function, f_y_function = assembler.build_f_components(mu1)

        f_x = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            self.fom_data["speed_reference_element_data"],
            f_x_function
        )

        f_y = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(
            self.fom_data["geometry_utilities"],
            self.fom_data["mesh"],
            self.fom_data["mesh_geometric_data"],
            self.fom_data["speed_dofs_data"],
            self.fom_data["speed_reference_element_data"],
            self.fom_data["speed_reference_element_data"],
            f_y_function
        )

        f_S = np.concatenate([
            f_x,
            f_y,
            np.zeros(self.fom_sol["pressure_n_dofs"])
        ])

        return self.assemble_reduced_vector(V, f_S)

    def create_POD_matricies(self,
                                solver,
                                p_boundary_info,
                                u_boundary_info,
                                speed_n_dofs,
                                inner_product_u,
                                B_1,
                                B_2,
                                tol=1e-6,
                                N_max=20
                                ):
        
        snapshot_matrix_u = []
        snapshot_matrix_s = []
        snapshot_matrix_p = []

        print("Performing FOM training...")

        n_train = len(self.training_set)

        for it, (mu0, mu1) in enumerate(self.training_set):
            print(
                f"[Training] sample {it + 1:03d}/{n_train} | "
                f"mu0={mu0:.4g}, mu1={mu1:.4g}"
            )

            fom_result = solver.solve_FOM(
                p_boundary_info=p_boundary_info,
                u_boundary_info=u_boundary_info,
                mu0=mu0,
                mu1=mu1,
                newton_tol=tol,
                max_iterations=N_max
            )

            if isinstance(fom_result, tuple):
                solution = fom_result[0]
            else:
                solution = fom_result

            snapshot = solution["u"]

            snapshot_u = snapshot[0:2 * speed_n_dofs]
            snapshot_p = snapshot[2 * speed_n_dofs:]
            snapshot_s = scipy.sparse.linalg.spsolve(inner_product_u,np.transpose(B_1 + B_2) @ snapshot_p)

            snapshot_matrix_u.append(snapshot_u)
            snapshot_matrix_p.append(snapshot_p)
            snapshot_matrix_s.append(snapshot_s)

        snapshot_matrix_u = np.array(snapshot_matrix_u)
        snapshot_matrix_p = np.array(snapshot_matrix_p)
        snapshot_matrix_s = np.array(snapshot_matrix_s)

        C_u = snapshot_matrix_u @ inner_product_u @ np.transpose(snapshot_matrix_u)
        C_s = snapshot_matrix_s @ inner_product_u @ np.transpose(snapshot_matrix_s)
        C_p = snapshot_matrix_p @ np.transpose(snapshot_matrix_p)

        return snapshot_matrix_u, snapshot_matrix_s, snapshot_matrix_p, C_u, C_s, C_p

    def assemble_supremizer_matricies(self,
                                      A_op,
                                      B_x_op,
                                      B_y_op,
                                      speed_n_dofs,
                                      pressure_n_dofs):
        
        X_1 = make_np_sparse(A_op.operator_dofs, [2 * speed_n_dofs, 2 * speed_n_dofs], [0, 0])
        X_2 = make_np_sparse(A_op.operator_dofs, [2 * speed_n_dofs, 2 * speed_n_dofs], [speed_n_dofs, speed_n_dofs])
        B_1 = make_np_sparse(B_x_op.operator_dofs, [pressure_n_dofs, 2 * speed_n_dofs], [0, 0])
        B_2 = make_np_sparse(B_y_op.operator_dofs, [pressure_n_dofs, 2 * speed_n_dofs], [0, speed_n_dofs])

        return X_1 + X_2, B_1, B_2
    
    def assemble_reduced_matrix(self, basis, fom_matrix):
        return basis.T @ fom_matrix @ basis

    def assemble_reduced_vector(self, basis, fom_vector):
        return basis.T @ fom_vector
    
    def eig_analysis(self, C, N_max=None, tol=1e-9):
        eigenvalues, eigenvectors_matrix = np.linalg.eigh(C)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order].real
        eigenvectors_matrix = eigenvectors_matrix[:, order].real

        eigenvalues = np.maximum(eigenvalues, 0.0)
        total_energy = np.sum(eigenvalues)

        if total_energy <= 0.0:
            raise ValueError("POD covariance matrix has zero total energy.")

        relative_retained_energy = np.cumsum(eigenvalues) / total_energy
        target_energy = 1.0 - tol
        N = np.argmax(relative_retained_energy >= target_energy) + 1

        if N_max is not None:
            N = min(N, N_max)

        eigenvectors = [eigenvectors_matrix[:, i] for i in range(N)]

        return N, eigenvectors
    
    def create_basis_functions_matrix(self, N, snapshot_matrix, eigenvectors, inner_product=None):
        basis_functions = []
        
        for n in range(N):
            eigenvector =  eigenvectors[n]
            basis = np.transpose(snapshot_matrix)@eigenvector
            if inner_product is not None:
                norm = np.sqrt(np.transpose(basis) @ inner_product @ basis) ## metti inner product
            else:
                norm = np.sqrt(np.transpose(basis) @ basis)

            basis /= norm
            basis_functions.append(np.copy(basis))

        basis_function_matrix = np.transpose(np.array(basis_functions))
        
        return basis_function_matrix