from torch.utils.data import Dataset
import inspect
import numpy as np
import torch

class MolecularDataset(Dataset):
    """Create a dataset with flexible number of objects
    One of the objects must be called "input" and it 
    will be used in the loss calculation during the during 
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # TODO: check if "input" was defined
        # The [0] dimension of all arguments is the batch size
        self.lenght = value.shape[0]
        
    def __len__(self):
        return self.lenght

    def __getitem__(self,idx):
        item_values = {}
        # getmembers() returns all the members of an object 
        for attribute in inspect.getmembers(self):
            # to remove private and protected functions
            if not attribute[0].startswith('_'):
                #to remove other methods and properties
                if isinstance(attribute[1],np.ndarray):
                    #get the values of all the attributes of the object
                    item_values[attribute[0]] = getattr(self,attribute[0])[idx]
                elif torch.is_tensor(attribute[1]):
                    #get the values of all the attributes of the object
                    item_values[attribute[0]] = getattr(self,attribute[0])[idx]
                   
        return item_values

def training_indices(N_samples, train_amount):
    indices = np.random.permutation(N_samples) 
    train_idx = int(train_amount * N_samples) 
    return indices[0:train_idx], indices[train_idx:N_samples]

def format_for_dataset(input_data, shape, idx, device):
    formatted_data = torch.reshape(torch.tensor(input_data), shape)
    formatted_data = formatted_data.float().to(device)
    return formatted_data[idx]

def periodic_dataset_forces(mol_sys, train_amount, embedding_property, device):
    N_samples = mol_sys.n_frames
    N_particles = mol_sys.n_particles_tot
    train_idx, test_idx = training_indices(N_samples, train_amount)

    train_dataset = MolecularDataset(coords = format_for_dataset(mol_sys.coords, (N_samples, N_particles, -1), train_idx, device),
                                     input = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), train_idx, device),
                                     box = format_for_dataset(mol_sys.box[:,0,:,:], (N_samples, 3, 2), train_idx, device),
                                     embedding_property = torch.tensor(embedding_property[train_idx]).to(device))
                                     # Setting the chosen feature as the labels for training
    test_dataset = MolecularDataset(coords = format_for_dataset(mol_sys.coords, (N_samples, N_particles, -1), test_idx, device),
                                     input = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), test_idx, device),
                                     box = format_for_dataset(mol_sys.box[:,0,:,:], (N_samples, 3, 2), test_idx, device),
                                     embedding_property = torch.tensor(embedding_property[test_idx]).to(device))

    return train_dataset, test_dataset

def not_periodic_dataset_forces(mol_sys, train_amount, embedding_property, device):
    N_samples = mol_sys.n_frames
    N_particles = mol_sys.n_particles_tot
    train_idx, test_idx = training_indices(N_samples, train_amount)

    train_dataset = MolecularDataset(coords = format_for_dataset(mol_sys.coords, (N_samples, N_particles, -1), train_idx, device),
                                     input = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), train_idx, device),
                                     embedding_property = torch.tensor(embedding_property[train_idx]).to(device))
                                     # Setting the chosen feature as the labels for training
    test_dataset = MolecularDataset(coords = format_for_dataset(mol_sys.coords, (N_samples, N_particles, -1), test_idx, device),
                                     input = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), test_idx, device),
                                     embedding_property = torch.tensor(embedding_property[test_idx]).to(device))

    return train_dataset, test_dataset

def periodic_dataset_mapping(mol_sys, feature, train_amount, device):
    N_samples = mol_sys.n_frames * mol_sys.n_molecules
    N_particles = mol_sys.n_particles_mol
    train_idx, test_idx = training_indices(N_samples, train_amount)

    train_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), train_idx, device),
                                     forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), train_idx, device), 
                                     box = format_for_dataset(mol_sys.box, (N_samples, 3, 2), train_idx, device))
                                     # Setting the chosen feature as the labels for training
    test_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), test_idx, device),
                                     forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), test_idx, device), 
                                     box = format_for_dataset(mol_sys.box, (N_samples, 3, 2), test_idx, device))

    return train_dataset, test_dataset

def not_periodic_dataset_mapping(mol_sys, feature, train_amount, device):
    N_samples = mol_sys.n_frames * mol_sys.n_molecules
    N_particles = mol_sys.n_particles_mol
    train_idx, test_idx = training_indices(N_samples, train_amount)

    train_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), train_idx, device),
                                     forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), train_idx, device))
                                     # Setting the chosen feature as the labels for training
    test_dataset = MolecularDataset(input = format_for_dataset(feature, (N_samples, N_particles, -1), test_idx, device),
                                    forces = format_for_dataset(mol_sys.forces, (N_samples, N_particles, 3), test_idx, device))
                                  
    return train_dataset, test_dataset
