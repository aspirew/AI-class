from collections import namedtuple
from functools import partial
from random import random, sample, choices, randint, randrange, shuffle
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

def roulette(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )
    
def tournament(population: Population, fitness_func: FitnessFunc) -> Population:
    size = randint(2, len(population)-1)
    selected_population = population.copy()
    shuffle(selected_population)

    selected_population = sorted(
        selected_population[0:size],
        key = lambda genome: fitness_func(genome),
        reverse = True
    )
    return selected_population[0:2]

def k_point_crossover(a: Genome, b: Genome, k: int) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError("Genomes are of a different size")
    
    length = len(a)
    if length < k - 1 or k < 1:
        raise ValueError("Invalid k size")
    
    offspring_1 = []
    offspring_2 = []

    crossover_points = sorted(sample(population = range(1, length), k=k))
    
    last_index = 0
    skip = False
    
    for index in crossover_points:
        if not skip:
            offspring_1 = offspring_1[:last_index] + a[last_index:index] + b[index:]
            offspring_2 = offspring_2[:last_index] + b[last_index:index] + a[index:]
        else:
            offspring_1 = offspring_1[:last_index] + a[last_index:]
            offspring_2 = offspring_2[:last_index] + b[last_index:]
        last_index = index
        skip = not skip
    if len(crossover_points) % 2 == 0:
        temp_a = offspring_1.copy()
        offspring_1 = offspring_1[0:last_index] + offspring_1[last_index:]
        offspring_2 = offspring_2[0:last_index] + temp_a[last_index:]
        
    return offspring_1, offspring_2

def flip_mutation(genome: Genome, num: int = 1, probability: float = MUTATION_PROBABILITY) -> Genome:
    for _ in range(num):
        if(random() < probability):
            index = randrange(len(genome))
            genome[index] = abs(genome[index] - 1)
    return genome

def inversion_mutation(genome: Genome, num: int = 1, probability: float = MUTATION_PROBABILITY) -> Genome:
    length = len(genome)
    for _ in range(num):
        if random() < probability:
            p1 = randint(1, length - 1)
            p2 = randint(p1, length - 1)
            inversed_part = genome[p1:p2]
            inversed_part.reverse()
            genome = genome[0:p1] + inversed_part + genome[p2:]

    return genome
    
    
def run(
    populate_func: PopulationFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    elitism: bool = True,
    selection_func: SelectionFunc = roulette,
    crossover_func: CrossoverFunc = k_point_crossover,
    mutation_func: MutationFunc = flip_mutation,
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
        
        next_generation = []

        if elitism:
            next_generation = population[0:2]
        
        for _ in range(int((len(population) - len(next_generation)) / 2)):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1], 1)
            offspring_a = mutation_func(offspring_a)
            offspring_a = inversion_mutation(offspring_a)
            offspring_b = mutation_func(offspring_b)
            offspring_b = inversion_mutation(offspring_b)
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

for _ in range(0, 1000):
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
    
average = average / 1000
print(average)