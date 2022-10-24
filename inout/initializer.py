import os, shutil
from datetime import datetime
from molecular_system.molecular_system import MolecularSystem, PeriodicMolecularSystem
from molecular_system.mapping_utils import *
from inout.file_reader import InputParser, MolecularFilesReader
 
class Initializer():
    """Class to read the input file and initialize the variables 
    necessary for a specific task. Each task is implemented as 
    a separate method:
    - initialize_for_mapping
    - initialize_for_force_matching
    - ...

    Auxiliary methods allow to handle initialization from specific
    trajectory formats. Currently only LAMMPS format is supported:
    - initialize_from_lammps
    """

    def __init__(self, filename):
        """Call InputParser to read the input file and store 
        parameter names and values in self.parameters.        
        """
        self.parameters = InputParser().parse(filename)
        self.reader = MolecularFilesReader()

    def initialize_for_mapping(self, cmdl_args):
        """ Reads the atomistic trajectory and properties from a 
        LAMMPS dump and data files and initializes a corresponding 
        PeriodicMolecularSystem object with the read values:
         - coordinates, images, box
         - forces
         - number of atoms, molecules, frames
         - masses, atom types, molecule membership
         - bonds list
                    
        Returns
        -------
        atom_sys: PeriodicMolecularSystem
            Initialized PeriodicMolecularSystem corresponding to the atomistic trajectory
        feature: 
            Selected feature for the mapping model: cooridinates or full distances matrix
        parameters: Parameters
            A dictionary-like object of model parameters and settings read from the input file
        """
        parameters = self.overwrite_from_command_line(self.parameters, cmdl_args)
        if not hasattr(parameters, 'overwrite'):
            parameters["overwrite"] = "no"

        atom_sys = self.initialize_from_lammps(parameters)
               
        parameters["shuffle"] = True # Shuffle the instances during training
        parameters["Tstart"] = 4 # Starting temperature in Gumbel-Softmax annealing
        parameters["molname"] = parameters["trajectory"].partition(".")[0] # Used for output file names
        parameters["task"] = "mapping"

        parameters = self.create_output_folder(parameters) 

        return atom_sys, parameters

    def initialize_for_force_matching(self, cmdl_args):
        """ Initializes the atomistic and coarse grained system objects
        with the attributes read from the input files.
        
        Parameters
        ----------
        input_file_name: str
            name of the input file for a force matching task
            Expected format of the input file: header line, path_to_data, 
            coords_and_forces_file, assignment_and_type_template_file, 
            mass_template_file, n_frames, n_atoms_per_mol, n_mol, n_cg_per_mol
        logger: 
            a logger object to output details of the systems
        Returns
        -------
        atom_sys, cg_sys: MolSystem()
            Initialized MolSystem objects corresponding to the atomistic and 
            mapped coarse grained trajectories
        """

        parameters = self.overwrite_from_command_line(self.parameters, cmdl_args)
        if not hasattr(parameters, 'overwrite'):
            parameters["overwrite"] = "no"

        atom_sys = self.initialize_from_lammps(parameters) 

        # Instanciate the mapped CG trajectory and fill it with data calculated from the atomistic one
        if eval(parameters["periodic"]) == True:
            cg_sys = PeriodicMolecularSystem(int(parameters["n_CG_mol"]), 
                                             int(parameters["n_molecules"]), 
                                             int(parameters["n_frames"]))
        else:
            cg_sys = MolecularSystem(int(parameters["n_CG_mol"]), 
                                     int(parameters["n_molecules"]), 
                                     int(parameters["n_frames"]))
        
        bead_idx = self.reader.read_file(parameters["directory"] + parameters["assignments"])
        del bead_idx[0] #remove header
        mapper = AtomisticToCGMapper()
        mapper.map_trajectory(atom_sys, cg_sys, bead_idx)

        parameters["shuffle"] = True # Shuffle the instances during training
        parameters["molname"] = parameters["trajectory"].partition(".")[0] # Used for output file names
        parameters["task"] = "force_matching"

        parameters = self.create_output_folder(parameters) 

        return atom_sys, cg_sys, parameters

    def initialize_for_simulation(self, cmdl_args):
        sim_parameters = self.overwrite_from_command_line(self.parameters, cmdl_args)
        if not hasattr(sim_parameters, 'overwrite'):
            sim_parameters["overwrite"] = "no"
        
        loaded_model = torch.load(sim_parameters["directory"] + sim_parameters["model_path"])
        parameters = loaded_model["parameters"]
        parameters["overwrite"] = sim_parameters["overwrite"]
        parameters["n_frames"] = 1

        atom_sys = self.initialize_from_lammps(parameters)
         # Instanciate the mapped CG trajectory and fill it with data calculated from the atomistic one
        if eval(parameters["periodic"]) == True:
            cg_sys = PeriodicMolecularSystem(int(parameters["n_CG_mol"]), 
                                             int(parameters["n_molecules"]), 
                                             int(parameters["n_frames"]))
        else:
            cg_sys = MolecularSystem(int(parameters["n_CG_mol"]), 
                                     int(parameters["n_molecules"]), 
                                     int(parameters["n_frames"]))
        
        bead_idx = self.reader.read_file(parameters["directory"] + parameters["assignments"])
        del bead_idx[0] #remove header
        mapper = AtomisticToCGMapper()
        mapper.map_trajectory(atom_sys, cg_sys, bead_idx)

        parameters["molname"] = parameters["trajectory"].partition(".")[0] # Used for output file names
        parameters["task"] = "simulation"

        parameters = self.create_output_folder(parameters) 

        return atom_sys, cg_sys, loaded_model, parameters, sim_parameters

    def initialize_from_lammps(self, parameters):
        """Read from LAMMPS data and dump files and store the 
        properties in a PeriodicMolecularSystem object.
        NOTE: Only systems with one molecule type supported.

        Arguments:
        ---------
        parameters: dict
            Dictionary with filenames and import parameters read 
            from the input file. It should containy:
            - n_particles_mol: number of particles of one molecule
            - n_molecules: number of molecules in the system
            - n_frames: number of frames to read. They will be taken 
                        from the beginning of the file
            - directory: relative path where the lammps files are located
            - trajectory: name of the lammps dump file
            - data_file: name of the lammps data file

        Returns:
        -------
        mol_sys: PeriodicMolecularSystem
            Object initialized with all properties read from the lammps files
        
        """
        # Instanciate the atomistic trajectory and fill it with data from the LAMMPS dump file
        n_particles_mol = int(parameters["n_atoms_mol"])
        n_molecules = int(parameters["n_molecules"])
        n_frames = int(parameters["n_frames"])

        if eval(parameters["periodic"]) == True:
            mol_sys = PeriodicMolecularSystem(n_particles_mol, n_molecules, n_frames)
        else:
            mol_sys = MolecularSystem(n_particles_mol, n_molecules, n_frames)
        self.reader.read_dump(parameters["directory"]+parameters["trajectory"],mol_sys)  
        self.reader.read_data(parameters["directory"]+parameters["data_file"],mol_sys) 
        return mol_sys 
   
    def overwrite_from_command_line(self, parameters, cmdl_args):
        i = 0
        for attribute in dir(cmdl_args):
            if not attribute.startswith('_'):  # exclude attributes that do not come from the input file
                cmd_value = getattr(cmdl_args, dir(cmdl_args)[i])
                if cmd_value is not None:
                    parameters[attribute] = str(cmd_value) 
            i=i+1                   
        return parameters

    def create_output_folder(self, parameters):
        # Overwrite the content in the default folder
        if eval(parameters["overwrite"]) == True:
            res_folder = parameters["directory"] + "default_output_" + parameters["task"] + "/"
            parameters["res_folder"] = res_folder
            if not os.path.exists(res_folder): #create the directory if it does not exists
                os.mkdir(res_folder)    
            elif os.listdir(res_folder):            # make a temporary backup if the directory 
                bak_folder = res_folder+"bak/"      # exists and it is not empty
                if os.path.exists(bak_folder): # remove the bak directory if it exists
                    shutil.rmtree(bak_folder)  
                shutil.copytree(res_folder, bak_folder)            

        # Create a new folder for the run
        elif eval(parameters["overwrite"]) == False:
            timestamp = str(datetime.utcnow().strftime('%Y_%m_%d_%H%M%S'))
            res_folder = parameters["directory"] + parameters["task"] + "_" + timestamp + "/"
            parameters["res_folder"] = res_folder
            if not os.path.exists(res_folder):
                os.mkdir(res_folder)   
            
        return parameters