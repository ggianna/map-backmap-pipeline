import numpy as np
import torch

class BackMapper():

     def __init__(self, latent_dim_back_map, original_dim):
        self.latent_dim_back_map = latent_dim_back_map
        self.original_dim = original_dim 
        if self.latent_dim_back_map > self.original_dim:
            raise ArithmeticError("The latent dimension cannot be higher than the original dimension.")
         

     def back_mapping(self, latent_config_vector):
         return np.pad(latent_config_vector, self.original_dim - self.latent_dim_back_map) # TODO: Fix this

     def get_latent_dim(self):
        return self.latent_dim_back_map

     def get_original_dim(self):
        return self.original_dim





# Unit test
if __name__ == "__main__":
   # Example of use
   myBackMapper = BackMapper(2, 10)
   myMappedResult = myBackMapper.back_map(torch.tensor(np.array([1,2])))
   print(myMappedResult)
   ################