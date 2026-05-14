import numpy as np
import scipy

from other_utilities import make_np_sparse

class ROM_Methods(object):
    def __init__(self,fom_sol,operators,training_set):
        self.training_set = training_set
        self.operators = operators
        self.fom_sol = fom_sol

    def solve_POD_Galerkin(self):
        pass

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
            "basis_functions_p": basis_functions_p
            }

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

            solution = solver.solve_FOM(
                                        p_boundary_info = p_boundary_info,
                                        u_boundary_info = u_boundary_info,
                                        mu0=mu0,
                                        mu1=mu1,
                                        newton_tol=tol,
                                        max_iterations=N_max
                                        )

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
    
    def eig_analysis(C, N_max=None, tol=1e-9):
        L_e, VM_e = np.linalg.eig(C)
        eigenvalues = []
        eigenvectors = []

        for i in range(len(L_e)):
            eig_real = L_e[i].real
            eig_complex = L_e[i].imag
            assert np.isclose(eig_complex, 0.)
            eigenvalues.append(eig_real)
            eigenvectors.append(VM_e[i].real)

        total_energy = sum(eigenvalues)
        retained_energy_vector = np.cumsum(eigenvalues)
        relative_retained_energy = retained_energy_vector/total_energy

        if all(flag==False for flag in relative_retained_energy>= tol) and N_max != None:
            N = N_max
        else:
            N = np.argmax(relative_retained_energy >= tol) + 1
        
        return N, eigenvectors
    
    def create_basis_functions_matrix(N, snapshot_matrix, eigenvectors, inner_product=None):
        basis_functions = []
        
        for n in range(N):
            eigenvector =  eigenvectors[n]
            basis = np.transpose(snapshot_matrix)@eigenvector
            if inner_product!= None:
                norm = np.sqrt(np.transpose(basis) @ inner_product @ basis) ## metti inner product
            else:
                norm = np.sqrt(np.transpose(basis) @ basis)

            basis /= norm
            basis_functions.append(np.copy(basis))

        basis_function_matrix = np.transpose(np.array(basis_functions))
        
        return basis_function_matrix