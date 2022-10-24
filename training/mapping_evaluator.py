import torch.nn as nn
import numpy as np
import networkx as nx

class MappingEvaluator():
    def __init__(self):
        nn.Module.__init__(self)

    def is_acceptable(self, encoder, connectivity):
        R_effective_CG = encoder.rounded_effective_CG()
        R_effective_CG = R_effective_CG.numpy()
        noise = 0
        signal = 0
        molecule_bond_matrix = connectivity.bond_matrix[0]
        atom_ids = np.nonzero(R_effective_CG)

        for moiety in range(len(R_effective_CG)):
            
            idx = atom_ids[1][np.where(atom_ids[0]==moiety)]
            in_moiety_connectivity_matrix = molecule_bond_matrix[np.ix_(idx,idx)]
            moiety_graph = nx.convert_matrix.from_numpy_matrix(in_moiety_connectivity_matrix)
            path = nx.shortest_path(moiety_graph)
            this_signal = sum([len(path[i])-1 for i in range(len(path))])
            signal = signal + this_signal
            max_signal = idx.size*(idx.size-1)
            this_noise = max_signal - this_signal
            noise = noise + this_noise
        
        if noise == 0:
            acceptable = True
        else:
            acceptable = False
        
        return acceptable
    
    def is_unique(self, new_mapping, old_mappings):
        mat_size = sum(new_mapping.size())
        mat1 = np.zeros([mat_size, mat_size])
        ncg1 = new_mapping.size()[0]
        naa1 = new_mapping.size()[1]
        mat1[ncg1:ncg1+ncg1,ncg1:ncg1+naa1] = np.array(new_mapping)
        G1 = nx.from_numpy_array(mat1)

        result = []
        for i in range(len(old_mappings)):
            mat_size = sum(old_mappings[i].size())
            mat2 = np.zeros([mat_size, mat_size])
            ncg2 = old_mappings[i].size()[0]
            naa2 = old_mappings[i].size()[1]
            mat2[ncg2:ncg2+ncg2,ncg2:ncg2+naa2] = np.array(old_mappings[i])
            G2 = nx.from_numpy_array(mat2)

            result.append(not nx.is_isomorphic(G1, G2))
        
        unique = all(i for i in result)
        return unique


