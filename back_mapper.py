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