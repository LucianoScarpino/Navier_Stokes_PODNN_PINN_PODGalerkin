import numpy as np

from pypolydim import polydim, gedim
from pypolydim.export_vtk_utilities import ExportVTKUtilities

from other_utilities import export_folder
from Discretization import Discretize
from Solver import Solver
from ROM import ROM_Methods

file_path, mesh_path, solution_path = export_folder("./Export")
reduced_model_path = file_path + "/Models/reduced_model.pkl"

mesh_size = 0.01
domain_area = 1.0
vertices = np.array([[0.0, 1.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0]])

discretizer = Discretize(vertices,mesh_size,domain_area,mesh_path)
solver = Solver(discretizer,solution_path)

info_internal,info_dirichlet,info_neumann_none = solver.set_dofs()

p_boundary_info = {
    0: info_internal,
    1: info_dirichlet,
    2: info_neumann_none,
    3: info_neumann_none,
    4: info_neumann_none,
    5: info_neumann_none,
    6: info_neumann_none,
    7: info_neumann_none,
    8: info_neumann_none
}

u_boundary_info = {
    0: info_internal,
    1: info_dirichlet,
    2: info_dirichlet,
    3: info_dirichlet,
    4: info_dirichlet,
    5: info_dirichlet,
    6: info_dirichlet,
    7: info_dirichlet,
    8: info_dirichlet
}

# Set FOM parameters
mu0 = 1.0
mu1 = 2.0
tol = 1.0e-6
max_it = 10

FOM_solution, FOM_Operators, FOM_data = solver.solve_FOM(p_boundary_info,
                                                u_boundary_info,
                                                mu0=mu0,
                                                mu1=mu1,
                                                newton_tol=tol,
                                                max_iterations=max_it,
                                                plot_solution=False)

# Set POD parameters
np.random.seed(26)

# Number of snapshots used to build POD
snapshot_num = 100

# MAximum number of POD retained
N_max = 100

mu0_range = [1., 10.]
mu1_range = [1., 3.]

P = np.array([mu0_range, mu1_range])
training_set = np.random.uniform(low=P[:, 0], high=P[:, 1], size=(snapshot_num, P.shape[0]))

# Offline
rom = ROM_Methods(FOM_solution,FOM_Operators,FOM_data,training_set=training_set)
reduced_elements = rom.reduce(solver,p_boundary_info,u_boundary_info,tol=tol,N_max=N_max)

# Save Reduced Model to run it online in -> "POD_online.py"
rom.save_reduced_model(
    reduced_elements,
    reduced_model_path,
    metadata={
        "snapshot_num": snapshot_num,
        "mu0_range": mu0_range,
        "mu1_range": mu1_range,
        "tol": tol,
        "N_max": N_max
    }
)

# Online
#mu0 = 2
#mu1 = 2
#rom_sol = rom.solve_POD_Galerkin(reduced_elements,mu0=mu0,mu1=mu1,newton_tol=tol,max_iterations=max_it)