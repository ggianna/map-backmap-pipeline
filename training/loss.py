from torch.nn.modules.loss import _Loss
import torch.nn as nn
import numpy as np
import math
import torch
import networkx as nx

class LossSelector():
    def __init__(self, parameters):
        """Extract parameters to pass to the loss function for initialization
        TODO: include here as well the encoder and connectivity for the last loss?
        """
        self.loss_selector = parameters["loss_selector"]
        self.forces_weight = float(parameters["forces_weight"])
    
    def select_function(self, encoder, connectivity):
        """Choose and initialize the loss function for the mapping task.
        Possible choices:
        - only_rec: reconstruction loss calculated using the Mean Squared Error    
                    loss = loss_rec
        - only_forces: mean squared force in the CG space loss
                    loss = loss_mf
        - rec_and_forces: reconstruction + mean squared force in the CG space loss
                    weighted according to the forces_weight parameter
                    loss = loss_rec + forces_weight * loss_mf
        - normal_rec_and_forces: as above, but normalized using the largest 
                    loss value seen until each epoch. 
                    forces_weight should be between 0 and 1
                    loss = loss_rec_norm * (1- forces_weight) + 
                                 forces_weight * loss_mf_norm
        - normal_rec_forces_connect: as above, but with added penalty if 
                    unconnected atoms are assigned to the same moiety.
                    loss_connect is computed using a Noise to Signal definition
                    loss = oss_rec_norm * (1- forces_weight) + 
                                 forces_weight * loss_mf_norm + loss_connect
        Returns:
        -------
        loss_function: the initialized loss function
        """
        if self.loss_selector == "only_rec":
            loss_function = ReconstructionLoss()
        elif self.loss_selector == "only_forces":
            loss_function = AverageForceLoss(encoder)
        elif self.loss_selector == "rec_and_forces":
            loss_function = RecAndForceLoss(encoder, self.forces_weight)
        elif self.loss_selector == "normal_rec_and_forces":
            loss_function = RecAndForceNormalizedLoss(encoder, self.forces_weight)
        elif self.loss_selector == "normal_rec_forces_connect":
            loss_function = RecForceConnectLoss(encoder, self.forces_weight, connectivity)
        else:    
            raise Exception("Loss function %s undefined", self.loss_selector) 
        return loss_function

class ReconstructionLoss(_Loss):
    def __init__(self):
        nn.Module.__init__(self)
        #Store the trend during training
        self.loss_reconstruction = []
        self.criterion = nn.MSELoss()

    def eval_loss(self, *args):
        args = args[0]
        predictions = args[0]
        batch = args[1]
        labels = batch["input"]
        loss_rec = self.criterion(predictions, labels)
        self.loss_reconstruction.append(loss_rec.detach())
        return loss_rec
    
    def reduce_loss(self,batch_size):
        old_loss = self.loss_reconstruction
        self.loss_reconstruction = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_reconstruction.append(np.mean(np.float_(old_loss[i:i+batch_size])))
            
class AverageForceLoss(_Loss):
    def __init__(self, encoder):
        nn.Module.__init__(self)
        #self.mapper = AtomisticToCGMapper()
        self.encoder = encoder
        #Store the trend during training
        self.loss_mean_forces = []

    def eval_loss(self, *args):
        args = args[0]
        batch = args[1]
        forces = batch["forces"].detach()
            # It is necessary to detach this from the graph otherwise 
            # the "retain_graph = True" error is triggered
        CG = self.encoder.CG()
        f0 = forces.reshape(-1, CG.size()[-1], 3)
        f = torch.matmul(CG, f0)   
        loss_mf = f.pow(2).sum(2).mean() #mean_force
        self.loss_mean_forces.append(loss_mf.detach())
        return loss_mf

    def get_mapper(self):
        return self.mapper

    def get_encoder(self):
        return self.encoder
    
    def reduce_loss(self,batch_size):
        old_loss = self.loss_mean_forces
        self.loss_mean_forces = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_mean_forces.append(np.mean(np.float_(old_loss[i:i+batch_size])))

class RecAndForceLoss(_Loss):
    def __init__(self, encoder, forces_weight):
        nn.Module.__init__(self)
        self.rec_loss = ReconstructionLoss()
        self.avg_force_loss = AverageForceLoss(encoder)
        self.forces_weight = forces_weight
        self.loss_reconstruction = []
        self.loss_mean_forces = []

    def eval_loss(self, *args):
        loss_rec = self.rec_loss.eval_loss(args[0])
        loss_mf = self.avg_force_loss.eval_loss(args[0])
        loss = loss_rec + loss_mf * self.forces_weight
        return loss
    
    def reduce_loss(self,batch_size):
        self.rec_loss.reduce_loss(batch_size)
        self.loss_reconstruction = self.rec_loss.loss_reconstruction
        self.avg_force_loss.reduce_loss(batch_size)
        self.loss_mean_forces = self.avg_force_loss.loss_mean_forces

class RecAndForceNormalizedLoss(_Loss):
    def __init__(self, encoder, forces_weight):
        nn.Module.__init__(self)
        self.rec_loss = ReconstructionLoss()
        self.avg_force_loss = AverageForceLoss(encoder)
        self.forces_weight = forces_weight
        self.unit_of_reconstruction_loss = 0
        self.unit_of_force_loss = 0
        self.loss_reconstruction = []
        self.loss_mean_forces = []

    def eval_loss(self, *args):
        loss_rec = self.rec_loss.eval_loss(args[0])
        loss_mf = self.avg_force_loss.eval_loss(args[0])
        max_loss_rec=float(max(self.unit_of_reconstruction_loss,loss_rec))
        max_loss_mf=float(max(self.unit_of_force_loss,loss_mf))
        self.unit_of_reconstruction_loss = max_loss_rec
        self.unit_of_force_loss = max_loss_mf
        loss_rec = loss_rec / max_loss_rec
        self.loss_reconstruction.append(loss_rec.detach())
        loss_mf = loss_mf / max_loss_mf
        self.loss_mean_forces.append(loss_mf.detach())
        loss = (1-self.forces_weight)*loss_rec + self.forces_weight * loss_mf 
        return loss

    def reduce_loss(self,batch_size):
        old_loss = self.loss_reconstruction
        self.loss_reconstruction = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_reconstruction.append(np.mean(np.float_(old_loss[i:i+batch_size])))

        old_loss = self.loss_mean_forces
        self.loss_mean_forces = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_mean_forces.append(np.mean(np.float_(old_loss[i:i+batch_size])))

class RecForceConnectLoss(_Loss):
    def __init__(self, encoder, forces_weight, connectivity):
        nn.Module.__init__(self)
        self.rec_and_force_norm_loss = RecAndForceNormalizedLoss(encoder, forces_weight)
        self.forces_weight = forces_weight
        self.unit_of_reconstruction_loss = 0
        self.unit_of_force_loss = 0
        self.connectivity = connectivity
        self.encoder = encoder
        #Store the trend during training   
        self.loss_connectivity = []   
        self.loss_reconstruction = []
        self.loss_mean_forces = []   
    
    def eval_loss(self, *args):
        loss_rec_forces = self.rec_and_force_norm_loss.eval_loss(args[0])
        loss_NtoS = self.compute_loss_NtoS(self.encoder)
        self.loss_connectivity.append(loss_NtoS)
        loss = loss_rec_forces + loss_NtoS
        return loss

    def compute_loss_NtoS(self, encoder):
        R_effective_CG = encoder.rounded_effective_CG()
        R_effective_CG = R_effective_CG.numpy()
        noise = 0
        signal = 0
        molecule_bond_matrix = self.connectivity.bond_matrix[0]
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
        
        loss_NtoS = self.NtoS(noise,signal)
        
        return loss_NtoS

    def NtoS(self, noise, signal):
        MIN_PENALTY = 3 
        return max(0.0, MIN_PENALTY + math.log10((10e-8 + noise)/(10e-8 + signal)))

    def reduce_loss(self,batch_size):
        self.rec_and_force_norm_loss.reduce_loss(batch_size)
        self.loss_reconstruction = self.rec_and_force_norm_loss.loss_reconstruction
        self.loss_mean_forces = self.rec_and_force_norm_loss.loss_mean_forces

        old_loss = self.loss_connectivity
        self.loss_connectivity = []
        for i in range(0,len(old_loss)-batch_size,batch_size):
            self.loss_connectivity.append(np.mean(np.float_(old_loss[i:i+batch_size])))

