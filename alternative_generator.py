from randomizer import Criterion
class AlternativeGenerator(Criterion):
    """A class for taking the restricted represantations of the input data and create approproate graphs."""
    def __init__(self,data):
        self.data = data


    def graph_data(self):
        print(self.data) 