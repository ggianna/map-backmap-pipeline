import inspect

class Parameters():
    """Class to store the parameters read from the input file"""
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def len(self):
        return len(self.__dict__)

    def merge(self, p1, p2):
        for attribute in dir(p1):
            if not attribute.startswith('_'):  # exclude attributes that do not come from the input file
                if not attribute.startswith('len'): # exclude the len attribute, if present
                    if not attribute.startswith('merge'):
                        setattr(self, attribute, p1[attribute])
        for attribute in dir(p2):
            if not attribute.startswith('_'):  # exclude attributes that do not come from the input file
                if not attribute.startswith('len'): # exclude the len attribute, if present
                    if not attribute.startswith('merge'):
                        setattr(self, attribute, p2[attribute])
                    

class HyperparametersGrid(Parameters):
    """Class to create a grid of values of the parameters read
    from the input file to perform hyperparameter optimization
    """
    def __init__(self):
        Parameters.__init__(self)
        raise NotImplementedError
