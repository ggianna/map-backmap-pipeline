import numpy as np
#x =  np.arange(2, 11).reshape(3,3)
#print(x)
import torch
torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""The Mapper maps a configuration representation represented as a vector
to a reduced (dimension) vector space.""" 
class Mapper():
            
      """Parameters
         ----------
         input_dim: Dimension of the feature vector of a single config representation 
         latent_dim: A compressed representation of the input data"""
      def __init__(self, input_dim, latent_dim):
         self.input_dim = input_dim
         self.latent_dim = latent_dim
         
         if self.input_dim < self.latent_dim:
            raise ArithmeticError("The input dimension cannot be higher than the latent dim")
         


      """Reduces the input feature vector - which represents a configuration - to the latent, reduced
      feature space.

      Parameters
      ----------
      config_feature_vector: The vector to map into the reduced space.
      """
      def mapping(self, config_feature_vector):
         return config_feature_vector[0:self.latent_dim]


      def get_input_dim(self):
         return self.input_dim

      def get_output_dimensions(self):
         return self.latent_dim   
        

# Unit test
if __name__ == "__main__":
   # Example of use
   myMapper = Mapper(10, 2)
   import numpy as np
   x = np.ones((3,3))
   print("Checkerboard pattern:")
   x = np.zeros((8,8),dtype=int)
   x[1::2,::2] = 1
   x[::2,1::2] = 1
   print(x)
   myMappedResult = myMapper.mapping(x)
   print(myMapper.get_input_dim())
   print(myMappedResult)
   ################
      
