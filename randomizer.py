from back_mapper import BackMapper
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


