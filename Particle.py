import numpy as np
import random
import math
from typing import List
from collections import namedtuple

Thing = namedtuple('Thing', ['value', 'weight'])

class Particle():

    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.personal_best_fitness = 0
        self.personal_best_position = [0]*len(position)

    def sigmoid(self, gamma):
        if gamma < 0:
            return 1 - 1/(1 + math.exp(gamma))
        else:
            return 1/(1 + math.exp(-gamma))

    def update_position(self):
        for i in range(len(self.position)):
            rnd = random.random()
            if (rnd < self.sigmoid(self.velocity[i])):
                self.position[i] = 1
            else:
                self.position[i] = 0

    def update_velocity(self, w, c1, c2, global_best_position):
        cognitive = c1 * random.random() * (np.asarray(self.personal_best_position) - np.asarray(self.position))
        social = c2 * random.random() * (np.asarray(global_best_position) - np.asarray(self.position))
        self.velocity = w * np.asarray(self.velocity) + cognitive + social
        
    def fitness(self, weight_limit: int, things: List[Thing]):       
        fitness = 0
        total_weight = 0
        
        for i, position_element in enumerate(self.position):
            fitness += position_element * things[i].value
            total_weight += position_element * things[i].weight
            
            if total_weight > weight_limit:
                return 0
        if(self.personal_best_fitness < fitness):
            self.personal_best_fitness = fitness
            self.personal_best_position = self.position 

        return fitness