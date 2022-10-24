def fittness_func():

    return 1




if __name__ == "__main__":
   # Example of use

   import pygad

   num_generations = 100
   num_parents_mating = 2
   fitness_func = fittness_func()
   initial_population = 2000
   sol_per_pop = 2
   num_genes = 28
   gene_type=float
   init_range_low = 0.5
   init_range_high = 5
   parent_selection_type="sss"
   keep_parents=-1
   keep_elitism=1
   K_tournament=3
   crossover_type="single_point"
   crossover_probability=None
   mutation_type="random"
   mutation_probability=None
   mutation_by_replacement=False
   mutation_percent_genes="default"
   mutation_num_genes=None
   random_mutation_min_val=-1.0
   random_mutation_max_val=1.0
   gene_space=None
   on_start=None
   on_fitness=None
   on_parents=None
   on_crossover=None
   on_mutation=None
   callback_generation=None
   on_generation=None


   ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       initial_population = initial_population,
                       sol_per_pop = sol_per_pop,
                       num_genes = num_genes,
                       gene_type=float,
                       init_range_low = init_range_low,
                       init_range_high = init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents = keep_parents,
                       crossover_type = crossover_type,
                       crossover_probability = crossover_probability,
                       mutation_type = mutation_type,
                       mutation_probability = mutation_probability,
                       mutation_by_replacement = mutation_by_replacement,
                       mutation_percent_genes = mutation_percent_genes,
                       mutation_num_genes = mutation_num_genes,
                       random_mutation_min_val = random_mutation_min_val,
                       random_mutation_max_val = random_mutation_max_val,
                       gene_space = gene_space,
                       on_start = on_start,
                       on_fitness = on_fitness,
                       on_parents = on_parents,
                       on_crossover = on_crossover,
                       on_mutation = on_mutation,
                       callback_generation = callback_generation,
                       on_generation = on_generation
                        )
                        
ga_instance.run()
ga_instance.plot_fitness()