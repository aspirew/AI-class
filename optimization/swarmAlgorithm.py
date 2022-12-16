from Particle import *
from collections import namedtuple

Thing = namedtuple('Thing', ['value', 'weight'])
INERTIA = 0.5
COGNITIVE = 1
SOCIAL = 10

particles: List[Particle] = []

def random_position(num_of_particles):
    position = []
    for _ in range(int(num_of_particles)):
        position.append(random.randint(0,1))
    return position

def random_velocity(num_of_particles):
    velocity = []
    for _ in range(int(num_of_particles)):
        velocity.append(random.uniform(-1,1))
    return velocity

def initialise(num_of_particles):
    particles.clear()
    for _ in range(num_of_particles):
        position = random_position(num_of_particles)
        velocity = random_velocity(num_of_particles)
        new_particle = Particle(position, velocity)
        particles.append(new_particle)

def run(w, c1, c2, used_things: List[Thing], num_of_particles, iteration_limit, fitness_limit, weight_limit):
    initialise(num_of_particles)

    global_best_fitness = 0
    global_best_position = [0]*int(num_of_particles)
    iteration = 0

    while (iteration < iteration_limit and global_best_fitness < fitness_limit):
        for p in particles:
            p.fitness(weight_limit, used_things)
            if(p.personal_best_fitness > global_best_fitness):
                global_best_position = p.personal_best_position
                global_best_fitness = p.personal_best_fitness
                # print(iteration, " -> ", global_best_fitness)
                
        for p in particles:
            p.update_velocity(w, c1, c2, global_best_position)
            p.update_position()

        iteration = iteration + 1
        
    return iteration
