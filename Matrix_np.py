from random import choice

import numpy as np

matrix = np.zeros((2, 2))
random_list = [0, 1]
for i in range(2):
    matrix[i] = [choice(random_list), choice(random_list)]

print(matrix)
