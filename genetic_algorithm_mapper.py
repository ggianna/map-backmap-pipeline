from re import L
from mapper import *
import numpy as np
from data_read import *

'''πρωτη αποσταση cc:1.54
   CH: οι πρωτες 6 τιμες 1.09, οι επομενες 6 2.1-2.2
   HH: 6 τιμες 1.77, 6 τιμες 2.5, 3 τιμες 3.1'''
class GeneticAlgorithmMapper(Mapper):

    def __init__(self, input_dim, latent_dim, data ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.data = data



        Mapper.__init__(self, self.input_dim, self.latent_dim)

    def create_list_of_distances_per_frame(self,distance_matrix_of_every_frame):

        list_of_lists = []
        b=0

        for i in range(distance_matrix_of_every_frame.shape[0]):
            list_of_each_frame = []
            #print(i)
            

            #list_of_each_frame = []
            for j in distance_matrix_of_every_frame[i,:][0]:
                #print("--------------------")
                #print(j)
                #print(type(j))
                #print("--------------------")
                
                #[take elements]  [of a specific frame]  [of a specific column] 
                #print(j)
                if b==8:
                    b=0
                j = j[b+1:8]
                for k in j:    #[take elements]  [of a specific frame]  [of a specific row] 
                    #print(k)
                    list_of_each_frame.append(k)  #data[1,:][0][1]

                b = b+1    
                  
            list_of_lists.append(list_of_each_frame)  

            if (len(list_of_lists) == 2000):

                return list_of_lists


        
        return list_of_lists
    
    def fitness(self,list_of_each_frame):

        #for CC bond we use the following function: h(x) = exp(-0.5*(x-Z)**2/S**2)/C
        # where Z               = 1.54056          
               #S               = 0.0297616        
               #C               = 7.5295


        # for CH bond we use F(x) = (fL(x) + fR(x))
	    #fL(x) = exp(-0.5*(x-YL)**2/SigL**2)/BL
	    #fR(x) = exp(-0.5*(x-YR)**2/SigR**2)/BR
        #where YR        = 2.17276          
        #SigR            = 0.0628838        
        #BR              = 31.4201          
        #YL              = 1.09359          
        #SigL            = 0.0318184       
        #BL              = 16.0178 


        # For HH bond we use W(x) = g1(x) + J(x)
	                        #g1(x) = exp(-0.5*(x-X1)**2/s1**2)/A1
	                        #J(x) = J21(x) + J22(x)
	                        #J21(x) = exp(-0.5*(x-X21)**2/s21**2)/A21
	                        #J22(x) = exp(-0.5*(x-X22)**2/s22**2)/A22

        # where  =======================            ==========================
        #X22             = 3.06293          
        #s22             = 0.0714689        
        #A22             = 89.1404          
        #X21             = 2.50901          
        #s21             = 0.124852         
        #A21             = 78.3694          
        #X1              = 1.77658          
        #s1              = 0.0606269        
        #A1              = 37.9529          
        list_of_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13]
        res = []
        for i in range(len(list_of_each_frame)):
            counter = 0
            list1 = []
            for j in list_of_each_frame[i]:
                if counter==0:
                    #print("We have CC bond")
                    Z = 1.54056 
                    S = 0.0297616
                    C = 7.5295

                    h = np.exp(-0.5*(j-Z)**2/S**2)/C
                    list1.append(h)
                    counter = counter+1


                elif(counter in list_of_numbers) :
                    #print("we have CH BOND")

                    YR = 2.17276          
                    SigR= 0.0628838        
                    BR= 31.4201          
                    YL= 1.09359          
                    SigL= 0.0318184       
                    BL= 16.0178

                    fL = np.exp(-0.5*(j-YL)**2/SigL**2)/BL
                    fR = np.exp(-0.5*(j-YR)**2/SigR**2)/BR
                    F = (fL + fR)
                    list1.append(F)
                    counter = counter +1


                else:

                    #print("We have HH BOND") 
                    X22= 3.06293          
                    s22= 0.0714689        
                    A22= 89.1404          
                    X21= 2.50901          
                    s21= 0.124852         
                    A21= 78.3694          
                    X1= 1.77658          
                    s1= 0.0606269        
                    A1= 37.9529 
                    J21 = np.exp(-0.5*(j-X21)**2/s21**2)/A21
                    J22 = np.exp(-0.5*(j-X22)**2/s22**2)/A22
                    
                    J = J21 + J22

                    g1 = np.exp(-0.5*(j-X1)**2/s1**2)/A1

                    W = g1 + J  
                    list1.append(W)
                    counter = counter +1 

            res.append(sum(list1))





        return res   



    def crossover_slice(self, list_of_probabilities,chromosomes):
        import random

        '''A function that chooses a random number between the length of the chromosome which is an element of the 2000x1x28 list.
           It takes the two chromosomes with the greater fitness value and picks a random position anx ex-changes the cells after that position between them'''

        '''First is to find the chromosomes with the greater fitness value.'''
        positions_of_two_greater_probabilities = []
        values_of_two_greater_probabilities = []

        
        for x in range(2):
            max_value = max(list_of_probabilities)
            max_index = list_of_probabilities.index(max_value)
            positions_of_two_greater_probabilities.append(max_index)
            values_of_two_greater_probabilities.append(max_value)
            list_of_probabilities.remove(max(list_of_probabilities))


        random_position_slice = random.randint(1,27)
        print("Random number from which i slice "+ str(random_position_slice))

        max_list1 = chromosomes[positions_of_two_greater_probabilities[0]]
        max_list2 = chromosomes[positions_of_two_greater_probabilities[1]]


        list_before_slicing1 = max_list1[0:random_position_slice]
        list_before_slicing2 = max_list2[0:random_position_slice]

        sliced_list1 = max_list1[random_position_slice:len(max_list1)]
        sliced_list2 = max_list2[random_position_slice:len(max_list2)]

        print("i am in crossover slice")

        print(max_list1)
        print(max_list2)

        print("--------------------------------")

        final_max_list1 = list_before_slicing1 + sliced_list2
        final_max_list2 = list_before_slicing2 + sliced_list1


        print("Crossovered max lists")

        print(max_list1)
        print(max_list2)

        print("--------------------------------")
            

        return final_max_list1, final_max_list2


    def crossover_slice_restricted(self,list_of_probabilities,chromosomes):

        import random

        '''A function that chooses a random number between the length of the chromosome which is an element of the 2000x1x28 list.
           It takes the two chromosomes with the greater fitness value and picks a random position anx ex-changes the cells after that position between them'''

        '''First is to find the chromosomes with the greater fitness value.'''
        positions_of_two_greater_probabilities = []
        values_of_two_greater_probabilities = []

        
        for x in range(2):
            max_value = max(list_of_probabilities)
            max_index = list_of_probabilities.index(max_value)
            positions_of_two_greater_probabilities.append(max_index)
            values_of_two_greater_probabilities.append(max_value)
            list_of_probabilities.remove(max(list_of_probabilities))


        random_positions = random.sample(range(1, 27), 2)
        #print(random_positions)

        max_list1 = chromosomes[positions_of_two_greater_probabilities[0]]
        max_list2 = chromosomes[positions_of_two_greater_probabilities[1]]


        list1_first_part = max_list1[0:min(random_positions)]
        list1_second_part = max_list1[max(random_positions):28]
        list2_first_part = max_list2[0:min(random_positions)]
        list2_second_part = max_list2[max(random_positions):28]

        

        sliced_list1 = max_list1[min(random_positions):max(random_positions)]
        sliced_list2 = max_list2[min(random_positions):max(random_positions)]

        print("i am in restricted slicing")

        print(max_list1)
        print(max_list2)

        print("-------------------------------")

        final_max_list1 = list1_first_part + sliced_list2 + list1_second_part
        final_max_list2 = list2_first_part + sliced_list1 + list2_second_part

        #print(final_max_list1)
        #print(final_max_list2)
        
            

        return final_max_list1, final_max_list2

    def random_crossover(self,list_of_probabilities,chromosomes):

        import random

        '''A function that chooses a random number between the length of the chromosome which is an element of the 2000x1x28 list.
           It takes the two chromosomes with the greater fitness value and picks a random position anx ex-changes the cells after that position between them'''

        '''First is to find the chromosomes with the greater fitness value.'''
        positions_of_two_greater_probabilities = []
        values_of_two_greater_probabilities = []

        
        for x in range(2):
            max_value = max(list_of_probabilities)
            max_index = list_of_probabilities.index(max_value)
            positions_of_two_greater_probabilities.append(max_index)
            values_of_two_greater_probabilities.append(max_value)
            list_of_probabilities.remove(max(list_of_probabilities))


        random_positions = random.choices(range(1, 27), k=random.randint(1,27))
        #print(random_positions)
        max_list1 = chromosomes[positions_of_two_greater_probabilities[0]]
        max_list2 = chromosomes[positions_of_two_greater_probabilities[1]]

        #print(max_list1)
        #print(max_list2)

        for i in random_positions:
            temp = max_list1[i]
            
            max_list1[i] = max_list2[i]

            max_list2[i] = temp



        print(max_list1)
        print(max_list2)

            

        return max_list1, max_list2  



    def remove_list_with_the_least_fitness(self,list_of_probabilities, chromosomes):
        # It removes the two chromosomes with the smaller fitness value and returns the final chromosome minus the two smaller chromosomes.
        positions_of_two_smaller_probabilities =[]
        values_of_two_smaller_probabilities = []

        for x in range(2):
            min_value = min(list_of_probabilities)
            min_index = list_of_probabilities.index(min_value)
            positions_of_two_smaller_probabilities.append(min_index)
            values_of_two_smaller_probabilities.append(min_value)
            #print(values_of_two_smaller_probabilities)
            list_of_probabilities.remove(min(list_of_probabilities))


        chromosomes.remove(chromosomes[positions_of_two_smaller_probabilities[0]]) 
        chromosomes.remove(chromosomes[positions_of_two_smaller_probabilities[1]])    

        return chromosomes  


    def mutation(self, chromosome1, chromosome2):
        import random
        '''A function that takes two crossovered chromosomes and picks with a 3/28 probability the genes of each chromosome to be changed.
           gene = 0.2*gene*u + 0.9*gene'''
        for i in range(len(chromosome1)):

            random_positions = random.randint(0,27)

            if (random_positions == 7 or random_positions == 9 or random_positions == 22):
                u = np.random.uniform()
                print(i)
                print(u)
                chromosome1[i] = 0.2*chromosome1[i]*u + 0.9*chromosome1[i]
                chromosome2[i] = 0.2*chromosome2[i]*u + 0.9*chromosome2[i]




        print("i am inside mutation")
        print(chromosome1)
        print(chromosome2)

        print("-----------------------------")


        return chromosome1, chromosome2



    def map(self, config_feature_vector):
        # Define a chromosome for a configuration
        # config_feature_vector: The result of the do_the mapping from main
        # e.g. chromosome = positions of atoms
        # Calculate the distances appropriately (all pairs)
        # In each frame we apply mutation to take random configurations of our input space and use the functionalities of pygad library

        # Generate a generation of new variations (individuals)

        # Evaluate the generation
        # Here we will use the distributions of distances that Dimitris shared (bonded and non-bonded pairwise distances)
        # So we will return as fitness the probability of the individual based on the above distributions
        # We must keep information about the type of the bond

        # After some iterations return the best individual
        #return super().map(config_feature_vector) # <== Change this

        


        return config_feature_vector

    def evaluateIndividual(self, individual):

        # Sum the probabilities of the distances across all pairs
        # return the value (or minus the value)
        #After we take the random representations of our input space, we use the fit functions that Dimitris sent.
        #We use a specific fit function based on the type of bond
        #We observe for each frame the configuration with the greater probability and then we select it for the final mapped list.

        return individual
    



if __name__ == "__main__":
    '''chromosome1, chromosome2= ga_mapper.random_crossover(ga_mapper.fitness(atom_sys),chromosomes)
    print("Chromosomes after random slicing")
    print(chromosome1)
    print(chromosome2)
    print("---------------------------------------------")

    mutated_chromosome1, mutated_chromosome2 = ga_mapper.mutation(chromosome1,chromosome2)

    print("Mutated Chromosomes after random slicing")
    print(mutated_chromosome1)
    print(mutated_chromosome2)
    print("---------------------------------------------")'''

    '''- prwta ypologizw fitness gia na brw ta 2 chromosomata me thn megaliteri pithanotita
   - meta efarmozw kapoio crossover
   - meta mutation sta 2 crossovered apotelesmata
   - meta afairw ta 2 chromosomata me thn mikroteri pithanotita
   - ksanaypologizw fitness
   - kai epanalamvanw mexri oses generations exw epileksei
   - telos ektipwnw to kalytero chromosoma kai sygkrinw me ayta pou eipe o dimitris'''
   # Example of use
   #reader = DataReader()
   #atom_sys = reader.read_data()
   #ga_mapper = GeneticAlgorithmMapper(8,3, atom_sys)
   #connect = Connectivity(atom_sys)
   #distances = connect.intramolecular_distances(atom_sys)
   #fit = ga_mapper.fitness(atom_sys)
   #print(len(fit))
   #print(fit)
   #chromosomes = ga_mapper.create_list_of_distances_per_frame(distances)
   #list1,list2 = ga_mapper.crossover_slice(ga_mapper.fitness(atom_sys),chromosomes)
   #print(list1)
   #print(list2)

   #chromosome= ga_mapper.remove_list_with_the_least_fitness(ga_mapper.fitness(atom_sys),chromosomes)
   #print(len(chromosome))
   
  
   #chromosome.append(list1)
   #chromosome.append(list2)
   #print(chromosome)
   #print(len(chromosome))



   

   #chromosome= ga_mapper.crossover_slice_restricted(ga_mapper.fitness(atom_sys),chromosomes)
   #print(len(chromosome))


   
  
   #chromosome.append(list1)
   #chromosome.append(list2)
   #print(chromosome)
   #print(len(chromosome))





    fitted_values = []

    num_generations = 30
    reader = DataReader()
    atom_sys = reader.read_data()
    ga_mapper = GeneticAlgorithmMapper(8,3, atom_sys)
    connect = Connectivity(atom_sys)
    distances = connect.intramolecular_distances(atom_sys)


    chromosomes = ga_mapper.create_list_of_distances_per_frame(distances)

    fit = ga_mapper.fitness(chromosomes)



    for i in range(num_generations):
       fitted_values.append(max(fit))
       print("Generation= " + str(i) )
       print("Fitness value for generation " + str(i)+" is "+str(max(fit)))
       #list1,list2 = ga_mapper.crossover_slice(fit,chromosomes)
       list1,list2 = ga_mapper.crossover_slice_restricted(fit,chromosomes)
       #list1,list2 = ga_mapper.random_crossover(fit,chromosomes)

       mutated_chromosome1, mutated_chromosome2 = ga_mapper.mutation(list1,list2)

       chromosomes.append(mutated_chromosome1)

       chromosomes.append(mutated_chromosome2)

       chromosomes = ga_mapper.remove_list_with_the_least_fitness(fit,chromosomes)

       fit = ga_mapper.fitness(chromosomes)

    from matplotlib import pyplot as plt   
    plt.plot(fitted_values)
    plt.show()


    