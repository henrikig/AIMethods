import numpy as np
import random

playerName = "myAgent"
nPercepts = 75  #This is the number of percepts
nActions = 7    #This is the number of actionss


# This is the class for your creature/agent
class MyCreature:

    def __init__(self):
        # Chromosome list of eleven weights with the following interpretation
        # 0: Eat strawberry
        # 1: Move closer to food
        # 2: Move away from food
        # 3: Move closer to bigger enemy
        # 4: Move away from bigger enemy
        # 5: Move closer to smaller enemy
        # 6: Move away from smaller enemy
        # 7: Move closer to equal sized enemy
        # 8: Move away from equal sized enemy
        # 9: Random move
        # 10: Do nothing
        self.chromosome = [random.random() for _ in range(11)]

    def AgentFunction(self, percepts):

        # Get direction to go based on row and column of target
        def _get_direction(row, col):
            directions = []
            # Move up
            if row < 2:
                directions.append(1)
            # Move down
            elif row > 2:
                directions.append(3)
            # Move left
            if col < 2:
                directions.append(0)
            # Move right
            elif col > 2:
                directions.append(2)

            return random.choice(directions)

        # Swap up with down and left with right if running away
        def _get_runaway_direction(dirct):
            a, b = (0, 2), (1, 3)
            if dirct in a:
                return a[1 - a.index(dirct)]
            elif dirct in b:
                return b[1 - b.index(dirct)]

        # Update actions based on chromosome and size of enemy
        def _handle_enemy(position, size, own_size, moves):
            # Handle bigger enemy
            if size > own_size:
                # Move closer to bigger enemy
                moves[position] += self.chromosome[3]
                # Move away from bigger enemy
                moves[_get_runaway_direction(position)] += self.chromosome[4]
            # Handle smaller enemy
            elif size < own_size:
                # Move closer to smaller enemy
                moves[position] += self.chromosome[5]
                # Move away from smaller enemy
                moves[_get_runaway_direction(position)] += self.chromosome[6]
            # Handle equal sized enemy
            else:
                # Move closer to equal sized enemy
                moves[position] += self.chromosome[7]
                # Move away from equal sized enemy
                moves[_get_runaway_direction(position)] += self.chromosome[8]

            return moves

        # Update actions based on chromosome and location of food
        def _handle_food(food_position, moves):
            # Move closer to food
            moves[food_position] += self.chromosome[1]
            # Move away from food
            moves[_get_runaway_direction(food_position)] += self.chromosome[2]

            return moves

        # 0 - move left
        # 1 - move up
        # 2 - move right
        # 3 - move down
        # 4 - do nothing
        # 5 - eat
        # 6 - move in a random direction
        actions = np.zeros(nActions)

        creature_map = percepts[:, :, 0]    # 5x5 map with information about creatures and their size
        food_map = percepts[:, :, 1]        # 5x5 map with information about strawberries
        wall_map = percepts[:, :, 2]        # 5x5 map with information about walls

        # Go through closest 8 squares.
        for i in range(1, 4):
            for j in range(1, 4):
                # If we are on our creature's current square and there's food, update action
                if i == j == 2:
                    if food_map[i, j] == 1:
                        actions[5] += self.chromosome[0]
                # Else, find position of tile and whether it contains food or an enemy
                else:
                    direction = _get_direction(i, j)
                    self_size = creature_map[2, 2]
                    enemy_size = creature_map[i, j]
                    food = food_map[i, j]

                    # Handle nearby enemy
                    if enemy_size < 0:
                        actions = _handle_enemy(direction, abs(enemy_size), self_size, actions)

                    # Handle nearby food
                    if food:
                        actions = _handle_food(direction, actions)

        actions[4] += self.chromosome[9]
        actions[6] += self.chromosome[10]

        return actions


def newGeneration(old_population):

    # Fitness function returns fitness based on number of rounds survived, size, and enemy and strawberry eats
    def _fitness(individual):
        f = individual.turn**(3/2)/10
        f += individual.size * 5
        f += individual.strawb_eats * 10
        f += individual.enemy_eats * 10
        f += individual.energy * 5
        individual.fitness = f
        return f

    # The num best creatures based on fitness are selected to the next generation
    def _elitism(new_pop, old_pop, num):
        for i in range(num):
            new_crt = MyCreature()
            new_crt.chromosome = old_pop[i].chromosome
            new_pop.append(new_crt)

        return new_pop

    # Parent selection with roulette wheel selection
    def _select_parent(probs, population):
        return population[np.random.choice(len(population), p=probs)]

    # The two selected parents' chromosomes are crossed to generate a new chromosome
    def _parent_crossover(p1, p2):
        i = random.choice(range(1, len(p1.chromosome)))
        new_chromosome = p1.chromosome[:i]
        new_chromosome += p2.chromosome[i:]
        return new_chromosome

    # Mutate one of the values in a chromosome to a new, random decimal
    def _mutate_chromosome(chromosome):
        n = random.choice(range(0, len(chromosome)))
        chromosome[n] = random.random()
        return chromosome

    # Fitness for all current agents
    N = len(old_population)
    fitness = np.zeros((N))

    # Calculate fitness for every individual
    for n, creature in enumerate(old_population):
        fitness[n] = _fitness(creature)

    # Sort population based on fitness for elitism
    old_sorted = sorted(old_population, key=lambda x: x.fitness, reverse=True)

    # Initialize new population
    new_population = list()

    # Get the three top performing agents from the old generation
    new_population = _elitism(new_population, old_sorted, 3)

    # Create values for roulette wheel selection
    total = sum([c.fitness for c in old_population])
    selection_probs = [c.fitness / total for c in old_population]

    # Create new children based on parents' fitness
    for i in range(len(old_population) - len(new_population)):
        parent1 = _select_parent(selection_probs, old_population)
        parent2 = _select_parent(selection_probs, old_population)

        child = MyCreature()

        child.chromosome = _parent_crossover(parent1, parent2)

        # Mutate with a small probability to ensure diversity
        if random.random() > 0.99:
            child.chromosome = _mutate_chromosome(child.chromosome)

        new_population.append(child)

    avg_fitness = np.mean(fitness)

    return (new_population, avg_fitness)
