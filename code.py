import numpy as np
x =  np.arange(2, 11).reshape(3,3)
#print(x)


class mapping():
    def __init__(self,data):
        self.data = data


class encoder(mapping):
    def __init__(self,data,input_dim, hidden_dim, latent_dim):
        super.__init__(data)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim



class dummy_mapper(mapping):   

    def convert(self,data):
        self.data = data
        return self




class back_mapping():
    def __init__(self,data):
        self.data = data



class decoder(back_mapping):
    def __init__(self,data,input_dim, hidden_dim, final_dim):
        super.__init__(data)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = final_dim


class dummy_back_mapper(back_mapping):

    def convert(self,data):
        self.data = data 
        return self     




class randomization(back_mapping):
    def __init__(self,data):
        self.data = data 


    def random_state(self):
        out_arr = np.dot(self.data,self.data)
        return out_arr





class criterion():

    def __init__(self,data):
        self.data = data



    def filtering():    
        def __init__(self,data):
            self.data = data




class graph():
    def __init__(self,data):
        self.data = data


    def graph_data(self):
        print(self.data)   
        
obj = dummy_back_mapper(x)
randomm = randomization(x)
print("test")
print(randomm.random_state())
print("ok")