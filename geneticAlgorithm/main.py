from collections import namedtuple
from functools import partial
from random import random, choices, randint, randrange
from typing import Callable, List, Tuple

Genome = List[int]
Population = List[Genome]
FitnessFunc = Callable[[Genome], int]
PopulationFunc = Callable[[],  Population]
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]]
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]]
MutationFunc = Callable[[Genome], Genome]
Thing = namedtuple('Thing', ['value', 'weight'])

# sums up to 1455 / 3648
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

POPULATION_SIZE = 10
WEIGHT_LIMIT = 3000
MAX_VALUE = 1000
MAX_WEIGHT = 1000
GENERATION_LIMIT = 1000
FITNESS_LIMIT = 1300
MUTATION_PROBABILITY = 1

def generate_thing() -> Thing:
    return Thing(randint(0, MAX_VALUE), randint(0, MAX_WEIGHT))

def generate_things(size: int) -> List[Thing]:
    return [generate_thing() for _ in range(size)]

def generate_genome(lenght: int) -> Genome:
    return choices([0, 1], k=lenght)

def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]

def fitness(genome: Genome, things: List[Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and things must be of same size")

    weight = 0
    value = 0
    
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value
            
            if weight > weight_limit:
                return 0            
    return value

def selection(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )
    
def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes are of a different size")
    
    length = len(a)
    if length < 2:
        return a, b
    
    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]

def mutation(genome: Genome, num: int = 1, probability: float = MUTATION_PROBABILITY) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if random() > probability else abs(genome[index] - 1)
    return genome
    
    
def run(
    populate_func: PopulationFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()
    
    for i in range(generation_limit):
        population = sorted(
            population,
            key = lambda genome: fitness_func(genome),
            reverse = True
        )
                        
        if fitness_func(population[0]) >= fitness_limit:
            break
        
        next_generation = population[0:2]
        
        for _ in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
            
        population = next_generation
        
    population = sorted(
        population,
        key = lambda genome: fitness_func(genome),
        reverse=True
    )
    
    return population, i

used_things = static_things
average = 0

for _ in range(0, 100):
    population, generations = run(
    populate_func=partial(
        generate_population, size=POPULATION_SIZE, genome_length=len(used_things)
    ),
    fitness_func=partial(
        fitness, things=used_things, weight_limit=WEIGHT_LIMIT
    ),
    fitness_limit=FITNESS_LIMIT,
    generation_limit=GENERATION_LIMIT
    )
    average = average + generations
    
average = average / 100
print(average)