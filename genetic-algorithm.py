#!/usr/bin/python

# import packages and modules
from multiprocessing.resource_sharer import stop
from pygad import GA
from numpy import abs, array, exp, sqrt, subtract
from numpy.linalg import norm
from time import time
from math import log10

#-----------------------------------

def main_genetic_algorithm(lower_range_value= -5.0, upper_range_value=5.0, mutation_values_pool=None, mut_probability=None):
    """Main function creating and running a genetic algorithm

    This function sets the values of the parameters for the constructor of 
    the pygad.GA class, runs the algorithm and plots the score of the 
    fittest solution with respect to the generation index.

    We have set:
    
    + number of generations produced: 100
    + number of parents/best-solutions used for mating: 2
    + fitness function: our on "home-made" one
    + number of solutions/chromosomes/frames per generation: 2000
    + number of genes/lengths per chromosome: 28

    No initial population is provided by us, so the constructor creates one
    by randomly picking up values from the integral: 
    [lower_range_value, upper_range_value]
    
    Parameters
    ----------
    lower_range_value: float
        Lowest value that could be picked up for the initial population

    upper_range_value: float
        Highest value that could be picked up for the initial population

    mutation_values_pool: list
        List containing the integrals from which values are picked for
        each respective gene, during its mutation. This could be a list
        of lists if some genes sould have their own pools of values.

    mut_probability: float
        Probability for each gene to be mutated
    """

    # Mandatory parameters
    num_generations = 30  # number of generations
    num_parents_mating = 2  # number of parents to mate
    #fitness_func = lengths_fitness_function  # user-defined fitness function
    fitness_func = coordinates_fitness_function  # (*)
    
    # Optional parameters
    initial_population = None  # user-specified initial population
    sol_per_pop = 500  # number of chromosomes/frames per generation
    #num_genes = 28  # number genes/bond-lengths per chromosome
    num_genes = 24  # (*)
    gene_type = float  # gene type
    init_range_low = lower_range_value  # lower value of random genes
    #init_range_low = -2.0  # (*)
    init_range_high = upper_range_value  # upper value of random genes
    #init_range_high = 2.0  # (*)
    parent_selection_type = "sss"  # parent selection type
    keep_parents = -1  # number of parents to be preserved
    keep_elitism = 0  # number of best solutions to be preserved
    crossover_type = "single_point"  # type of crossover operation
    crossover_probability = None  # probability to apply crossover to a parent
    mutation_type = "random"  # type of mutation operation
    mutation_probability = mut_probability  # probability of mutating a gene
    #mutation_by_replacement = True  # whether apply replacement (True) or summation (False) as mutation
    mutation_by_replacement = False  # (*)
    mutation_percent_genes = "default"  # = 10, percentage of genes to mutate
    mutation_num_genes = None  # number of genes to mutate

    if mutation_values_pool is None:
        random_mutation_min_val = -1.0  # lower number to be used in mutation process
        #random_mutation_min_val = -0.1  # (*)
        random_mutation_max_val = 1.0  # upper value to be used in mutation process
        #random_mutation_max_val = 0.1  # (*)
        gene_space = None  # set of values for each gene to be used in mutation process
    else:
        random_mutation_min_val = -1.0  # lower number to be used in mutation process
        random_mutation_max_val = 1.0  # upper value to be used in mutation process
        gene_space = mutation_values_pool

    on_start = None
    on_fitness = None
    on_parents = None
    on_crossover = None
    on_mutation = None
    callback_generation = None
    on_generation = None
    on_stop = None
    delay_after_gen = 0.0
    save_best_solutions = True  # save best solution in each generation in attribute best_solutions
    save_solutions = False
    suppress_warnings = False
    allow_duplicate_genes = True
    stop_criteria = None  # some criteria to stop the evolution
    parallel_processing = ['process', 10] # Was: None
    # TODO: Restore
    # random_seed = 1234567  #None  # (default)
    random_seed = None # (default)


    # creation of instance
    ga_instance = GA(
                     num_generations=num_generations,
                     num_parents_mating=num_parents_mating,
                     fitness_func=fitness_func,
                     initial_population=initial_population,
                     sol_per_pop=sol_per_pop,
                     num_genes=num_genes,
                     gene_type=gene_type,
                     init_range_low=init_range_low,
                     init_range_high=init_range_high,
                     parent_selection_type=parent_selection_type,
                     keep_elitism = keep_elitism,
                     keep_parents=keep_parents,
                     crossover_type=crossover_type,
                     crossover_probability=crossover_probability,
                     mutation_type=mutation_type,
                     mutation_probability=mutation_probability,
                     mutation_by_replacement=mutation_by_replacement,
                     mutation_percent_genes=mutation_percent_genes,
                     mutation_num_genes=mutation_num_genes,
                     random_mutation_min_val=random_mutation_min_val,
                     random_mutation_max_val=random_mutation_max_val,
                     gene_space=gene_space,
                     on_start=on_start,
                     on_fitness=on_fitness,
                     on_parents=on_parents,
                     on_crossover=on_crossover,
                     on_mutation=on_mutation,
                     callback_generation=callback_generation,
                     on_generation=on_generation,
                     on_stop=on_stop,
                     delay_after_gen=delay_after_gen,
                     save_best_solutions=save_best_solutions,
                     save_solutions=save_solutions,
                     suppress_warnings=suppress_warnings,
                     allow_duplicate_genes=allow_duplicate_genes,
                     stop_criteria=stop_criteria,
                     parallel_processing=parallel_processing,
                     random_seed = random_seed

                    )


    # start the optimization
    ga_instance.run()

    # access best solution found
    ga_instance.plot_fitness()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    

    # turn coordinates list into bond lengths list
    position_vectors_list = [array(solution[i:(i + 3)]) for i in range(0, len(solution), 3)]

    bond_lengths_list = []
    atoms_counted = 0
    for position_vector_1 in position_vectors_list[:-1]:
        for position_vector_2 in position_vectors_list[(atoms_counted + 1):]:
            bond_vector = subtract(position_vector_2, position_vector_1)
            bond_lengths_list.append(norm(bond_vector, 2))
        
        atoms_counted += 1
    print("Parameters of the best solution (coordinates) : {solution}".format(solution=solution)) 
    print("Parameters of the best solution (bond lengths) : {solution}".format(solution=bond_lengths_list))    
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    print(centers(solution))
    

    return None

#-----------------------------------

# create fitness function (2)
def coordinates_fitness_function(chromosome, chromosome_idx):
    """Fitness function receiving an atomistic representation of a frame
    and returning its fitness score based on its bond lengths and the PDFs.
    """
    
    
    number_of_atoms = int(len(chromosome) / 3.0)  # three coordinates per atom
    number_of_unique_bond_lengths = int((number_of_atoms**2 - number_of_atoms) / 2.0)

    # list containing the triplets of coordinates of each atom as arrays
    position_vectors_list = [array(chromosome[i:(i + 3)]) for i in range(0, len(chromosome), 3)]

    # turn coordinates list into bond lengths list
    bond_lengths_list = []
    atoms_counted = 0
    for position_vector_1 in position_vectors_list[:-1]:
        for position_vector_2 in position_vectors_list[(atoms_counted + 1):]:
            bond_vector = subtract(position_vector_2, position_vector_1)
            bond_lengths_list.append(norm(bond_vector, 2))
        
        atoms_counted += 1

    if len(bond_lengths_list) != number_of_unique_bond_lengths:
        print(f"Error: len(bond_lengths_list): {len(bond_lengths_list)} is not equal to " \
            + f"number_of_unique_bond_lengths: {number_of_unique_bond_lengths}!")
        quit()

    from numpy import linalg as LA
    center = centers(chromosome)

    distance_from_zero = LA.norm(center,2)    
    length_based_fitness = lengths_fitness_function(bond_lengths_list, chromosome_idx)
    distance_based_fitness = normalized_distance_fitness(distance_from_zero)
        
    #score = length_based_fitness + distance_based_fitness
    # TODO: Restore
    score = length_based_fitness + 5.0* distance_based_fitness

    # DEBUG LINES
    print("Current score: %5.2f (lengths)\t %5.2f (distance)"%(length_based_fitness, distance_based_fitness))
    #############

    return score

MIN_MEANINGFUL_DISTANCE = 1e-3

def normalized_distance_fitness(dist):
    dist = max(dist, MIN_MEANINGFUL_DISTANCE) # TODO: Examine if 10e-3 is appropriate
    return log10(1.0 / dist) / 3.0 # Since the max value of the ratio is 10^3, log10 maxes at 3, so normalization happens if we divide by 5

# create the fitness function (1)
def lengths_fitness_function(chromosome, chromosome_idx):
    score1, score2, score3 = 0.0, 0.0, 0.0
    for idx, gene in enumerate(chromosome):
        if idx == 0:
            score1 += PDF_CC(gene, 1.54056, 0.0297616, 7.5295)
        elif idx in range(1, 13):
            score2 += PDF_CH(gene, 2.17276, 0.0628838, 31.4201, 1.09359, 0.0318184, 16.0178)
        elif idx in range(14, 29):
            score3 += PDF_HH(gene, 3.06293, 0.0714689, 89.1404, 2.50901, 0.124852, 78.3694, 1.77658, 0.0606269, 37.9529)

    return (score1 + score2 + score3)
#    return (1.0 / (score1 + score2 + score3 - 0.95)**2)
#    return (1.0 / sqrt((score1 - 0.13)**2 + (score2 - 0.552)**2 + (score3 - 0.27)**2))
        

# probability density function for the C-C bond
def PDF_CC(r, Z, S, C):
    h = exp(-0.5 * (r - Z)**2 / S**2) / C

    return h

# probability density function for the C-H bonds
def PDF_CH(r, YL, SigL, BL, YR, SigR, BR):
    fL = exp(-0.5 * (r - YL)**2 / SigL**2) / BL
    fR = exp(-0.5 * (r - YR)**2 / SigR**2) / BR
    F = fL + fR
    
    return F

# probability density function for the H-H bonds
def PDF_HH(r, X21, s21, A21, X22, s22, A22, X1, s1, A1):
    J21 = exp(-0.5 * (r - X21)**2 / s21**2) / A21
    J22 = exp(-0.5 * (r - X22)**2 / s22**2) / A22
    g1 = exp(-0.5 * (r - X1)**2 / s1**2) / A1
    W = J21 + J22 + g1

    return W


import numpy as np #Function that returns each coordinate of the geometric center of a molecule 


def centers(chromosome):

    avg_coords = 0*[3]
    sum1= 0
    sum2= 0
    sum3 =0
    for i in range(0, len(chromosome), 3):
        sum1 = sum1 + chromosome[i]

    for j in range(1,len(chromosome),3):
        sum2 = sum2+chromosome[j]   


    for k in range(2,len(chromosome),3):
        sum3 = sum3+chromosome[k]

    x_coord = sum1/8  
    y_coord = sum2/8
    z_coord = sum3/8


    avg_coords.append(x_coord)
    avg_coords.append(y_coord)
    avg_coords.append(z_coord)


    return avg_coords
    


def center_of_mass(masses): # x,y,z,value
    import numpy
    nonZeroMasses = masses[numpy.nonzero(masses[:,3])]
    CM = numpy.average(nonZeroMasses[:,:3], axis=0, weights=nonZeroMasses[:,3]) 
    return CM  

#-----------------------------------

# 1x28 : [C1-C2, C1-H1, C1-H2, C1-H3, C1-H4, C1-H5, C1-H6, C1-H7, C1-H8,
#                C2-H1, C2-H2, C2-H3, C2-H4, C2-H5, C2-H6, C2-H7, C2-H8,
#                       H1-H2, H1-H3, H1-H4, H1-H5, H1-H6, H1-H7, H1-H8,
#                              H2-H3, H2-H4, H2-H5, H2-H6, H2-H7, H2-H8,
#                                     H3-H4, H3-H5, H3-H6, H3-H7, H3-H8,
#                                            H4-H5, H4-H6, H4-H7, H4-H8,
#                                                   H5-H6, H5-H7, H5-H8,
#                                                          H6-H7, H6-H8,
#                                                                 H7-H8]

mutation_values_pool = []
one_peak_CC = [(x / 10000.0) for x in range(14000, 17000)]
mutation_values_pool.append(one_peak_CC)

# Mutation pool (1)
#two_peaks_CH = [(x / 1000.0) for x in range(900, 2500)]
#for i in range(12):
#    mutation_values_pool.append(two_peaks_CH)
#three_peaks_HH = [(x / 1000.0) for x in range(1450, 3500)]
#for i in range(15):
#    mutation_values_pool.append(three_peaks_HH)

# Mutation pool (2)
first_peak_CH = [(x / 10000.0) for x in range(9000, 13000)]
second_peak_CH = [(x / 10000.0) for x in range(18200, 25000)]
for i in range(3):
    mutation_values_pool.append(first_peak_CH)
for i in range(6):
    mutation_values_pool.append(second_peak_CH)
for i in range(3):
    mutation_values_pool.append(first_peak_CH)
first_peak_HH = [(x / 10000.0) for x in range(14500, 20500)]
second_peak_HH = [(x / 10000.0) for x in range(21000, 28200)]
third_peak_HH = [(x / 10000.0) for x in range(28200, 35000)]
for i in range(2):
    mutation_values_pool.append(first_peak_HH)
mutation_values_pool.append(third_peak_HH)
for i in range(2):
    mutation_values_pool.append(second_peak_HH)
mutation_values_pool.append(first_peak_HH)
mutation_values_pool.append(second_peak_HH)
mutation_values_pool.append(third_peak_HH)
for i in range(3):
    mutation_values_pool.append(second_peak_HH)
mutation_values_pool.append(third_peak_HH)
for i in range(3):
    mutation_values_pool.append(first_peak_HH)

#-----------------------------------

t1 = time()
#main_genetic_algorithm(0.5, 5.0, mutation_values_pool, (1.0 / 28.0))
main_genetic_algorithm(0.5, 5.0)  # (*)
t2 = time()
print(f"time: {t2 - t1}")
#
