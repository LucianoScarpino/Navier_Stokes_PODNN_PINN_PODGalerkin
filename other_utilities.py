import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import matplotlib.tri
import os

from pypolydim import polydim

def make_np_sparse(A_sparse_data, new_size = None, shifts = None, transpose = None):
    if new_size is None:
        new_size = [A_sparse_data.size[0], A_sparse_data.size[1]]
    if shifts is None:
        shifts = [0, 0]
    if transpose is None:
        transpose = False
    if transpose:
        return scipy.sparse.csc_array((A_sparse_data.values, 
                                       ([i + shifts[1] for i in A_sparse_data.cols],
                                        [i + shifts[0] for i in A_sparse_data.rows])), 
                                      shape=(new_size[0], new_size[1]))
    else:
        return scipy.sparse.csc_array((A_sparse_data.values, 
                                       ([i + shifts[0] for i in A_sparse_data.rows],
                                        [i + shifts[1] for i in A_sparse_data.cols])), 
                                      shape=(new_size[0], new_size[1]))
                               
def plot_mesh(mesh, export_folder = ""):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_aspect('equal')
    
    coordinates = mesh.cell0_ds_coordinates()
    ax1.triplot(matplotlib.tri.Triangulation(coordinates[0, :], coordinates[1, :]), 'ko-', lw=1)
    ax1.grid(True)

    if export_folder != "":
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        file_name = 'Mesh.png'
        file_path = os.path.join(export_folder, file_name)
        plt.savefig(file_path)
        plt.show()
        plt.close(fig)
    else:
        plt.pause(0.1)
        plt.close(fig)

def evaluate_function_on_points(points, function_name):
    num_points = points.shape[1]
    function_values = np.zeros(num_points)

    for p in range(1, num_points):        
      function_values[p] = function_name(points[0, p], points[1, p], points[2, p])
    return function_values

def plot_solution(mesh, solution_cell0Ds, title = None, export_folder = None):
    plot_solution_on_coordinates(mesh.cell0_ds_coordinates(), solution_cell0Ds, title, export_folder)

def plot_solution_on_coordinates(coordinates, solution_on_coordinates, title = None, export_folder = None):
    if title is None:
        title = "Solution"
    if export_folder is None:
        export_folder = ""
    
    x = coordinates[0,:]
    y = coordinates[1,:]
    z = solution_on_coordinates
    triang = matplotlib.tri.Triangulation(x, y)
    
    fig = plt.figure(figsize = plt.figaspect(0.5))
    fig.suptitle(title)
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(triang, z, shading='flat')
    ax1.triplot(matplotlib.tri.Triangulation(coordinates[0, :], coordinates[1, :]), 'k--', lw=1)
    fig.colorbar(tpc)
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_trisurf(x, y, z, triangles=triang.triangles, cmap=plt.cm.Spectral)
    
    if export_folder != "": 
        if not os.path.exists(export_folder):
            os.makedirs(export_folder)
        file_name = title + '.png'
        file_path = os.path.join(export_folder, file_name)
        plt.savefig(file_path)
        plt.show()
    else:
        plt.show()

def export_folder(file_path):
    # Export Folder 
    export_file_path = file_path
    if not os.path.exists(export_file_path):
        os.makedirs(export_file_path)

    # Mesh file path
    export_mesh_path = export_file_path + "/Mesh"
    if not os.path.exists(export_mesh_path):
        os.makedirs(export_mesh_path)

    # Solution file path
    export_solution_path = export_file_path + "/Solution/FOM"
    if not os.path.exists(export_solution_path):
        os.makedirs(export_solution_path)

    return export_file_path,export_mesh_path,export_solution_path

def plot_FOM_solution(mesh,
                      speed_dofs_data,
                      u_x_numeric,
                      u_x_strong,
                      u_y_numeric,
                      u_y_strong,
                      pressure_dofs_data,
                      p_numeric,
                      p_strong,
                      vtk_utilities,
                      export_solution_path,
                      plot_path
                      ):
    

    u_x_on_cell0Ds = polydim.pde_tools.assembler_utilities.pcc_2_d.extract_solution_on_cell0_ds(
    mesh,
    speed_dofs_data,
    u_x_numeric,
    u_x_strong
    )

    u_y_on_cell0Ds = polydim.pde_tools.assembler_utilities.pcc_2_d.extract_solution_on_cell0_ds(
        mesh,
        speed_dofs_data,
        u_y_numeric,
        u_y_strong
    )

    p_on_cell0Ds = polydim.pde_tools.assembler_utilities.pcc_2_d.extract_solution_on_cell0_ds(
        mesh,
        pressure_dofs_data,
        p_numeric,
        p_strong
    )

    zero_cell0 = np.zeros_like(u_x_on_cell0Ds.numeric_solution)

    vtk_utilities.export_solution_2(export_solution_path + '/u_x',
                                mesh, 
                                u_x_on_cell0Ds.numeric_solution,
                                zero_cell0,
                                zero_cell0,
                                zero_cell0
                                )
    
    vtk_utilities.export_solution_2(export_solution_path + '/u_y',
                                    mesh, 
                                    u_y_on_cell0Ds.numeric_solution,
                                    zero_cell0,
                                    zero_cell0,
                                    zero_cell0
                                    )
    
    vtk_utilities.export_solution_2(export_solution_path + '/p',
                                    mesh, 
                                    p_on_cell0Ds.numeric_solution,
                                    zero_cell0,
                                    zero_cell0,
                                    zero_cell0
                                    )
    
    plot_solution(mesh, u_x_on_cell0Ds.numeric_solution, "u_x",export_folder=plot_path) 
    plot_solution(mesh, u_y_on_cell0Ds.numeric_solution, "u_y",export_folder=plot_path)
    plot_solution(mesh, np.sqrt(u_x_on_cell0Ds.numeric_solution * u_x_on_cell0Ds.numeric_solution +\
                        u_y_on_cell0Ds.numeric_solution * u_y_on_cell0Ds.numeric_solution), 
                        "u_mag",export_folder=plot_path)
    plot_solution(mesh, p_on_cell0Ds.numeric_solution, "p",export_folder=plot_path) 