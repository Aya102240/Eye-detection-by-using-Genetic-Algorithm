import numpy as np
from numpy.random import rand
import Image_with_circle
from random import randint

n_gene_ori = 1


# =====================================================================================================================

def matrix(rows, cols):
    my_matrix = [([0] * cols) for _ in range(rows)]
    return my_matrix


# =====================================================================================================================

def pop(sizeOfPop):
    popu = matrix(sizeOfPop, 1)
    for i in range(len(popu)):
        popu[i] = [randint(1, 31)]
    return popu


# =====================================================================================================================

def encode_pop(popu):
    my_matrix = matrix(len(popu), 5)
    for i in range(len(popu)):
        binary_num = bin(int(popu[i][0]))[2:].zfill(5)
        list_of_sting_binary_number = list(binary_num)
        bin1 = list(map(int, list_of_sting_binary_number))
        my_matrix[i] = bin1
    return my_matrix


# =====================================================================================================================

def decode_list(list_):
    my_num = int("".join(list(map(str, list_))), 2)
    return my_num


# =====================================================================================================================

def decode_pop(Encoded_pop):
    my_matrix = matrix(len(Encoded_pop), 1)

    for i in range(len(Encoded_pop)):
        my_matrix[i] = [int("".join(list(map(str, Encoded_pop[i]))), 2)]
    return my_matrix


# =====================================================================================================================

def fitness_fun(popu):
    quality = 1 / (Image_with_circle.desired_output(popu[0]) - Image_with_circle.unknowncircle(popu[0]))
    return quality


# =====================================================================================================================

def pop_fitness(popu):
    qualities_ = matrix(len(popu), 1)  # qualities is a matrix contain the number of rows in population
    for indv_num in range(len(popu)):
        qualities_[indv_num] = fitness_fun(
            popu[indv_num])  # for each row in qualities we assign the fitness of each pop(chromosome)
    return qualities_


# =====================================================================================================================

def select_mating_pool(popu, qualities_, num_parents, Num_of_genesInOriginal_pop):
    parents_ = matrix(num_parents, Num_of_genesInOriginal_pop)
    for parent_num in range(num_parents):
        max_qual_idx = np.where(qualities_ == np.max(qualities_))
        max_qual_idx = max_qual_idx[0][0]
        parents_[parent_num] = popu[max_qual_idx]
    return parents_


# =====================================================================================================================

def crossover(newParents, rate_crossover, n_individuals=6):
    newPopulation = matrix(6, 5)

    # previous parents (best parents).
    if rand() < rate_crossover:
        newPopulation[0: len(newParents)] = newParents
        newPopulation[3][0: 3] = newParents[0][0: 3]
        newPopulation[4][0: 3] = newParents[1][0: 3]
        newPopulation[4][3:] = newParents[2][3:]
        newPopulation[5][0: 3] = newParents[2][0: 3]
        newPopulation[5][3:] = newParents[0][3:]
    return newPopulation


# =====================================================================================================================

def mutation(Population, rate_mutation):
    if rand() < rate_mutation:
        for indx in range(len(Population)):
            if Population[indx][3] == 0:
                Population[indx][3] = 1
            elif Population[indx][3] == 1:
                Population[indx][3] = 0
    return Population


# =====================================================================================================================

# All definitions

# size of pop
gen_ = 50

# number of parents in mating pool
sol_per_pop = 6

# number of itration
iterate = 20

# number of genes per chromosome in genotype
num_genes = 1

# crossover rate
r_cross = 0.75

# mutation rate
r_mute = 0.15

# =====================================================================================================================


population = pop(gen_)

listQ = []
list_of_best = []
qualities = pop_fitness(population)
max_ = np.max(qualities)
listQ.append(max_)
idx = qualities.index(max_)
list_of_best.append(population[idx])

for iteration in range(iterate):
    # selecting the best parents in the population for mating
    parents = select_mating_pool(population, qualities, 3, n_gene_ori)
    # Generating next generation using crossover
    new_population = crossover(encode_pop(parents), r_cross, n_individuals=sol_per_pop)
    new_population = mutation(new_population, r_mute)
    # meaning the fitness of each chromosome in the population
    qualities = pop_fitness(new_population)
    DEcoded = decode_pop(new_population)
    print(f"Generation decoded{iteration}: ", DEcoded)

    max_ = np.max(qualities)
    listQ.append(max_)
    index_of_result_ = qualities.index(max_)
    idx = DEcoded[index_of_result_]
    list_of_best.append(idx)

max_ = np.max(listQ)
index_of_result = listQ.index(max_)
index_ = list_of_best[index_of_result]

rad_of_best = int(str(index_[0]))

Image_with_circle.draw_circle(rad_of_best - 22)
Image_with_circle.second_circle(rad_of_best - 22)
print(f"radius[{rad_of_best}] - 23: ", rad_of_best - 22)
