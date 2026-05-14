import numpy as np

from pypolydim import polydim, gedim
from pypolydim.export_vtk_utilities import ExportVTKUtilities

from other_utilities import export_folder
from Discretization import Discretize
from Solver import Solver

file_path, mesh_path, solution_path = export_folder("./Export/Test_1")

mesh_size = 0.001
domain_area = 1.0
vertices = np.array([[0.0, 1.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0]])

discretizer = Discretize(vertices,mesh_size,domain_area,mesh_path)
solver = Solver(discretizer)

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

u,u_x,u_y,p,iterations,rel_increment,converged = solver.solve_FOM(p_boundary_info,
                                                                  u_boundary_info,
                                                                  mu0=mu0,
                                                                  mu1=mu1,
                                                                  newton_tol=tol,
                                                                  max_iterations=max_it)