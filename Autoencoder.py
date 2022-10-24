#import numpy as np
#x =  np.arange(2, 11).reshape(3,3)
#print(x)
from turtle import forward
import torch
from mapper import *
from back_mapper import *
from data_read import *
import torch.nn.functional as F
from torch import Tensor

torch.cuda.is_available = lambda : False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AE( Mapper, BackMapper, torch.nn.Module):

      def map(self, config_feature_vector):
         # HINT: This will essentially call the forward function 
         # of the encoding part of the  NN
         reconstructed = forward(config_feature_vector)
         return reconstructed
      
      def back_map(self, latent_config_vector):

         # HINT: This will essentially call the forward function 
         # of the decoding part of the  NN
         reconstructed = forward(latent_config_vector)
         return reconstructed

      def train(self, model, criterion, optimizer, all_config_feature_vectors, epochs):
         # TODO: Complete
         #train_data_encoder = AE.map(all_config_feature_vectors,model)
         #train_data = AE.back_map(train_data_encoder)
         losses=[]  
         for i in range(epochs):  
            ypred=model.forward(all_config_feature_vectors)  
            loss=criterion(ypred, all_config_feature_vectors)  
            print("epoch:",i,"loss:",loss.item())  
            losses.append(loss)  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

      '''model = AE(input_shape=784).to(device)

         # create an optimizer object
         # Adam optimizer with learning rate 1e-3
         optimizer = optim.Adam(model.parameters(), lr=1e-3)

         # mean-squared error loss
         criterion = nn.MSELoss()'''
      
      def __init__(self, num_of_frames, num_of_molecules, num_of_atoms, num_of_coords_per_atom, num_of_latent_dimensions):
         
         # Store the provided arguments
         self.num_of_frames = num_of_frames
         self.num_of_molecules = num_of_molecules
         self.num_of_atoms = num_of_atoms
         self.num_of_coords_per_atom = num_of_coords_per_atom


         #self.input_dimension_encoder = self.num_of_atoms * self.num_of_coords_per_atom
         self.input_dimension_encoder = self.num_of_atoms
         self.output_dimension_encoder = num_of_latent_dimensions
         self.input_dimension_decoder = num_of_latent_dimensions
         self.output_dimension_decoder = self.input_dimension_encoder

         # Call multiple inits, based on all the classes I inherit from
         Mapper.__init__(self, self.input_dimension_encoder, self.output_dimension_encoder)
         BackMapper.__init__(self, self.input_dimension_decoder, self.output_dimension_decoder)
         torch.nn.Module.__init__(self)

         # DEBUG LINES
         from pprint import pprint
         print ("++Debugging AE")
         pprint(self.__dict__)
         print ("--Debugging AE")
         #############


                 
         # Building a linear encoder with Linear
        # layer followed by Relu activation function


         # Encoder
         # Flattened version
         self.enc1 = nn.Linear(in_features= self.input_dimension_encoder , out_features= self.output_dimension_encoder)

         # Decoder 
         self.dec1 = nn.Linear(in_features= self.input_dimension_decoder, out_features= self.output_dimension_decoder)

         # self.encoder = torch.nn.Sequential(
             #torch.nn.Linear(self.num_of_frames*self.num_of_molecules*self.input_dimension_encoder*self.coords, self.num_of_frames*self.num_of_molecules*6*self.coords),
             #torch.nn.ReLU(),
             #torch.nn.Linear(self.num_of_frames*self.num_of_molecules*6*self.coords, self.num_of_frames*self.num_of_molecules*self.output_dimension_encoder*self.coords),
             #torch.nn.ReLU()
        #)
         
         #self.decoder = torch.nn.Sequential(
          #   torch.nn.Linear(self.num_of_frames*self.num_of_molecules*self.input_dimension_decoder*self.coords, self.num_of_frames*self.num_of_molecules*6*self.coords),
           #  torch.nn.ReLU(),
            # torch.nn.Linear(self.num_of_frames*self.num_of_molecules*6*self.coords, self.num_of_frames*self.num_of_molecules*self.output_dimension_decoder*self.coords),
             #torch.nn.ReLU()
        #)


      def forward(self,x):
         
         x = F.relu(self.enc1(x))

         x = F.relu(self.dec1(x))
         #encoded = self.encoder(x)
         #decoded = self.decoder(encoded)

         return x
         #return decoded

# Unit test
if __name__ == "__main__":
   # Example of use
   reader = DataReader()
   # DEBUG LINES
   # print(reader)
   #############
   atom_sys = reader.read_data()
   # DEBUG LINES
   # print(atom_sys.coords)
   # print(atom_sys.coords.shape[0])
   #############

   # myMapper = Mapper(atom_sys.coords.shape[2], torch.tensor(2))
   # myBackMapper = BackMapper(torch.tensor(2), atom_sys.coords.shape[2]) 
   numberOfMoieties = 2
   ae = AE(atom_sys.coords.shape[0], atom_sys.coords.shape[1], atom_sys.coords.shape[2], atom_sys.coords.shape[3], numberOfMoieties)

   #myMappedResult = myMapper.map(torch.tensor(np.array([1,2,3,4,5,6,7,8,9,0])))
   # DEBUG LINES
   print(ae)
   print(ae.back_map(ae.coords))
   #############

   ################
