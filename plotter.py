import matplotlib.pyplot as plt
import numpy as np

class plotter():
    """Initialize an object from the latent dimensions that is for decoded.
    It is the reverse process of the mapping and we ended up with a single solution, which
    we are going to produce alternatives2. 
    """
    def __init__(self,data):
        self.data = data


    def plot(self,data):
        #plt.plot(data)
        #plt.xlabel("x axis") 
        #plt.ylabel("y axis") 
        #plt.show()
        fig = plt.figure(figsize=(4,4))

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(data[0],data[1],data[2]) 

        plt.show()


