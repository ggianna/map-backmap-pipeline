#################################################
"""
Comments 

"""
#################################################

# Set profiling on
#from memory_profiler import profile

# Imports

import os
import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
import torch.nn as nn
from training.trainer import ModelTrainer 
from inout.logger import Logger
from inout.initializer import Initializer
from inout.renderer import Renderer
from inout.argparser import init_parser
from training.scheduler import TScheduler
from models.mapping_model import *
from models.layers import *
from inout.renderer import Renderer
from inout.plotter import Plotter
from training.dataset import *
from training.mapping_evaluator import MappingEvaluator
from inout.file_writer import MolecularFilesWriter
from molecular_system.connectivity import Connectivity



import torch
#torch.cuda.is_available = lambda : False
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#@profile
#torch.version.cuda

class DataReader:

    def __init__(self):
        super().__init__()

    def read_data(self):
        #################################################
        # Global settings

        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

        # Set our seeds
        SEED = 2345
        if SEED is not None:
            np.random.seed(SEED)
            #random.seed(SEED)
            torch.manual_seed(SEED)
            torch.manual_seed(SEED)
            torch.backends.cudnn.deterministic = True
            os.environ['PYTHONHASHSEED'] = str(SEED)


        #################################################
        # Parse command line args
        parser = init_parser("mapping")
        cmdl_args = parser.parse_args()

        if cmdl_args.input_file is None:
            input_file = 'force_matching.in'
            #input_file = 'mapping.in'
        else:
            input_file = cmdl_args.input_file

        #################################################
        # Input & Preprocessing

        #atom_sys, parameters = Initializer(input_file).initialize_for_mapping(cmdl_args)
        atom_sys, cg_sys, parameters = Initializer(input_file).initialize_for_force_matching(cmdl_args)
        connect = Connectivity(atom_sys)
        feature = connect.intramolecular_distances(atom_sys) #mou dinei tis apostaseis sto atomisitc level

        #print(atom_sys.coords.shape[2])
        #print("---------------------------------------")
        #print(cg_sys.coords[0])
        #print("---------------------------------------")
        #print(feature[0])

        return atom_sys
    #print(torch.__version__)
    #frames =  torch.tensor(2000)
    #num_of_molecules = torch.tensor(1)
    #input_dim = torch.tensor(8)
    #latent_dim = torch.tensor(2)
    #xyz = torch.tensor(3)


    #my_mapper = AE(atom_sys, num_of_molecules, input_dim, latent_dim, xyz)
    #print(my_mapper)







    '''data_for_encoding = EncoderMapper(atom_sys.coords,2000,1,latent_dim,3)

    n_ep = 8
    b_s = 124
    l_r = 2e-2
    model = data_for_encoding.map()

    crit = nn.MSELoss()
    opti = torch.optim.AdamW(
    model.parameters(), lr=l_r)
    for ep in range(n_ep):

        result= model(atom_sys.coords)
        loss = crit(result, atom_sys.coord)
        loss.backward()
        opti.step()
        opti.zero_grad()
        print(result, 'epoch_n [{epoch + 1},{n_ep}], loss of info:{loss.info.item()}') '''
    

    #################################################
    # Parse command line args
    #parser = init_parser("mapping")
    #cmdl_args = parser.parse_args()

    #if cmdl_args.input_file is None:
     #   input_file = 'mapping.in'
    #else:
     #   input_file = cmdl_args.input_file

    #################################################
    # Input & Preprocessing

    #atom_sys, parameters = Initializer(input_file).initialize_for_mapping(cmdl_args)
    #connect = Connectivity(atom_sys)
    #connect.bond_matrix = connect.create_bond_matrix(atom_sys.bonds_list)  # Move elsewhere?
    # Select the input feature for the mapping model TODO: make this a nn.Module
    #if parameters["feature"]=="distances":
     #   feature = connect.intramolecular_distances(atom_sys)
    #elif parameters["feature"]=="coordinates":
     #   feature = atom_sys.coords
    #################################################################


# Run main function, if appropriate
if __name__ == "__main__":
    reader = DataReader()
    print(reader)
    atom_sys = reader.read_data()
    #print(data[2,:][0][1]) # 3o frame 2h grammh
    #print(data.coords[0].shape)    
    #print(data.coords[0])
    #print(type(data.types))
    print(atom_sys.coords)
