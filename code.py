import numpy as np
x =  np.arange(2, 11).reshape(3,3)
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




class BackMapper():
    """Initialize an object from the latent dimensions that is for decoded. 
    """
    def __init__(self,data):
        self.data = data



class Decoder(BackMapper):
    def __init__(self,data,input_dim, hidden_dim, final_dim):
        """Parameters
           ----------
           input_dim Dimension of our input data(after encoder):, 
           hidden_dim:Number of hidden layers,
           final_dim: The original dimensions of our input data before encoding"""
        super.__init__(data)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = final_dim


class DummyBackMapper(BackMapper):
    """It was created for familirization purposes.
       It takes the input data and returns the same"""
    def convert(self,data):
        self.data = data 
        return self     




class Randomizer(BackMapper):
    """A class for creating random represantations of the decoded data like rotations"""
    def __init__(self,data):
        self.data = data 


    def random_state(self):
        """It was created for familirization purposes.
           It takes the input data and multiply them by their self to create a random result  """      
        out_arr = np.dot(self.data,self.data)
        return out_arr





class Criterion(Randomizer):
    """A class for defining criterions to restrict the random represantations
       from the randomizer."""
    def __init__(self,data):
        self.data = data



    def filter():  
        def __init__(self,data):
            self.data = data




class AlternativeGenerator(Criterion):
    """A class for taking the restricted represantations of the input data and create approproate graphs."""
    def __init__(self,data):
        self.data = data


    def graph_data(self):
        print(self.data)   
        
obj = dummy_back_mapper(x)
randomm = randomization(x)
print("test")
print(randomm.random_state())
print("ok")