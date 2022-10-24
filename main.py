from mapper import *
from back_mapper import *
import numpy as np
import torch
from Autoencoder import *
from data_read import *
from genetic_algorithm_mapper import *
def main():
    reader = DataReader()
    print(reader)
    atom_sys = reader.read_data()
    print(atom_sys.coords[0][:,0])
    numberOfMoieties = 2
    # mapper = Mapper(data.coords.shape[2],torch.tensor(3))
    # back_mapper= BackMapper(torch.tensor(3),data.coords.shape[2])
    ae = AE(atom_sys.coords.shape[0], atom_sys.coords.shape[1], atom_sys.coords.shape[2], atom_sys.coords.shape[3], numberOfMoieties)

    mapped_data = do_the_mapping(ae, atom_sys.coords)
    back_mapped_data = do_the_back_mapping(ae, mapped_data)
    print(mapped_data[0])

    ga_mapper = GeneticAlgorithmMapper(atom_sys.coords.shape[2], numberOfMoieties, mapped_data)
    print(len(ga_mapper.map(ga_mapper.data)))


    # TODO: actually training
    # do_the_mapping(ae, ae)

def do_the_mapping(mapper, data):
    # Read data
    print(mapper.input_dim)
    print(mapper.latent_dim)    
    # Map all data and print output
    # Init result list
    mapped_data = []
    # For each frame
    #print("Eimai sto do the mapping kai num of frames = " +str(mapper.num_of_frames))
    for frame in range(mapper.num_of_frames):
        #print(frame)
        # call the mapper
        mapped_frame = mapper.mapping(data[frame])


        # Add to result list
        mapped_data.append(mapped_frame)

    # Backmap all data
    # For each frame

    back_mapped_data = []
    print(len(mapped_data))
    print(mapped_data[1999].shape)

    return mapped_data


def do_the_back_mapping(back_mapper, mapped_data):

    back_mapped_data = []

    for frame in mapped_data:
        # call the back_mapper
        backmapped_frame = back_mapper.back_mapping(frame)
        #print(backmapped_frame)


        back_mapped_data.append(backmapped_frame)
        # Print output and deviation from original positions
        # TODO: Complete
    return back_mapped_data    

if __name__ == "__main__":
    main()
