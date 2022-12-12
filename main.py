import swarmAlgorithm
import geneticAlgorithm
from collections import namedtuple
import random

Thing = namedtuple('Thing', ['value', 'weight'])
MAX_VALUE = 1000
MAX_WEIGHT = 1000
WEIGHT_LIMIT = 3000
ITERATION_LIMIT = 1000
FITNESS_LIMIT = 1310

# sums up to 1405 / 3148
static_things = [
    Thing(500, 2200),
    Thing(150, 160),
    Thing(100, 70),
    Thing(500, 200),
    Thing(10, 38),
    Thing(5, 25),
    Thing(40, 333),
    Thing(15, 80),
    Thing(60, 350),
    Thing(30, 192),
]

fitness = 0
   
for _ in range(30):
    fitness = fitness + swarmAlgorithm.run(swarmAlgorithm.INERTIA, swarmAlgorithm.COGNITIVE, swarmAlgorithm.SOCIAL, static_things, len(static_things), ITERATION_LIMIT, FITNESS_LIMIT, WEIGHT_LIMIT)

fitness = fitness / 30
print(fitness)