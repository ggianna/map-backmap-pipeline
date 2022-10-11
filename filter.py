from back_mapper import BackMapper

class GraphDistanceCriterion(BackMapper):
    """A class for creating random represantations of the decoded data like rotations"""
    def __init__(self,data):
        self.data = data 


    def filter():  
        def __init__(self,data):
            self.data = data


class FilteredDummyBackMapper(BackMapper):
    """It was created for familirization purposes.
       It takes the input data and returns a random output of the data"""
    def criterion(self,data,criterion):
        """Criterion is something getting an input which is a back_mapped thing and answering with true or false"""
        sum = 0
        for i in range(len(data)):
            for j in range(len(data)):
                sum = sum + data[i][j]

        if sum<50:
            return 1

        else:  
            return 0   