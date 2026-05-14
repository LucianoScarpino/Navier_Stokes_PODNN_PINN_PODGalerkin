from pypolydim import polydim

from Assembler import Assembler

class Solver(object):
    def __init__(self,discretizer:object):
        self.method_type,self.mesh,\
            self.mesh_connectivity_data,self.mesh_geometric_data,self.geometry_utilities = discretizer.discretize()
        
    def solve(self,p_boundary_info:dict,u_boundary_info:dict,mu0:float,mu1:float,method:str='PODGalerkin'):
        pressure_reference_element_data,speed_reference_element_data,\
                    pressure_mesh_dofs_info,pressure_dofs_data,\
                            speed_mesh_dofs_info,speed_dofs_data,\
                                 speed_n_dofs,pressure_n_dofs,tot_dofs = self.Taylor_Hood_FEM(self.method_type,
                                                                                        self.mesh,
                                                                                        self.mesh_connectivity_data,
                                                                                        p_boundary_info,
                                                                                        u_boundary_info)
        configuration = self.geometry_utilities,self.mesh,\
                            self.mesh_geometric_data,speed_dofs_data,\
                                speed_reference_element_data,speed_mesh_dofs_info,\
                                    pressure_dofs_data,pressure_mesh_dofs_info,pressure_reference_element_data
        
        assembler = Assembler(configuration=configuration)

        J_S, F, u_x_strong, u_y_strong, p_strong = assembler.assemble_linear_system(speed_n_dofs=speed_n_dofs,
                                                                                    pressure_n_dofs=pressure_n_dofs,
                                                                                    tot_dofs=tot_dofs,
                                                                                    mu0=mu0,
                                                                                    mu1=mu1
                                                                                    )
        
        

    def set_dofs(self):
        info_internal = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
        info_internal.marker = 0

        info_dirichlet = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.strong)
        info_dirichlet.marker = 1

        info_neumann_none = polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo(polydim.pde_tools.do_fs.DOFsManager.MeshDOFsInfo.BoundaryInfo.BoundaryTypes.none)
        info_neumann_none.marker = 0

        return info_internal,info_dirichlet,info_neumann_none
    
    def Taylor_Hood_FEM(self,method_type,mesh,mesh_connectivity_data,p_boundary_info,u_boundary_info):
        pressure_reference_element_data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type, 1)
        speed_reference_element_data = polydim.pde_tools.local_space_pcc_2_d.create_reference_element(method_type,2) 
        dof_manager = polydim.pde_tools.do_fs.DOFsManager()
        pressure_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(pressure_reference_element_data, 
                                                                                            mesh, 
                                                                                            p_boundary_info)
        pressure_dofs_data = dof_manager.create_do_fs_2_d(pressure_mesh_dofs_info, 
                                                        mesh_connectivity_data)

        speed_mesh_dofs_info = polydim.pde_tools.local_space_pcc_2_d.set_mesh_do_fs_info(speed_reference_element_data, 
                                                                                        mesh, 
                                                                                        u_boundary_info)
        speed_dofs_data = dof_manager.create_do_fs_2_d(speed_mesh_dofs_info, 
                                                    mesh_connectivity_data)
        
        pressure_n_dofs = pressure_dofs_data.number_do_fs
        pressure_n_strongs = pressure_dofs_data.number_strongs
        speed_n_dofs = speed_dofs_data.number_do_fs
        speed_n_strongs = speed_dofs_data.number_strongs
        tot_dofs = 2 * speed_n_dofs + pressure_n_dofs
        tot_strongs = 2 * speed_n_strongs + pressure_n_strongs

        print('-'*60)
        print("P dofs\t", "P stgs\t", "U dofs\t", "U stgs\t", "T dofs\t", "T stgs")
        print(pressure_n_dofs,"\t", pressure_n_strongs,"\t", speed_n_dofs,"\t", speed_n_strongs,"\t", tot_dofs,"\t", tot_strongs)
        print('-'*60)
        
        return pressure_reference_element_data,speed_reference_element_data,\
                pressure_mesh_dofs_info,pressure_dofs_data,\
                speed_mesh_dofs_info,speed_dofs_data,\
                speed_n_dofs,pressure_n_dofs,tot_dofs