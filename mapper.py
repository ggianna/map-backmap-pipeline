#import numpy as np
#x =  np.arange(2, 11).reshape(3,3)
#print(x)


class Mapper():
    """Initialize an object which will be encoded
    """
    def __init__(self,data):
        self.data = data


class Encoder(Mapper):
     """Is a feed-forward neural network that is structured to predict
        the latent view representation of the input data"""       
     def __init__(self,data,input_dim, hidden_dim, latent_dim):
        """Parameters
           ----------
           input_dim Dimension of our input data(distance matrix):, 
           hidden_dim:Number of hidden layers,
           latent_dim: A compressed representation of the input data"""
        super.__init__(data)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim



class DummyMapper(Mapper):   
    """It was created for familirization purposes.
        It takes the input data and returns the same"""
    def convert(self,data):
        self.data = data
        return self




#obj = dummy_back_mapper(x)
#randomm = randomization(x)
#print("test")
#print(randomm.random_state())
#print("ok")