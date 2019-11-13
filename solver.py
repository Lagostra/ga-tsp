"""
GA-based TSP solver following this tutorial: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
"""

import numpy as np
import random
import operator
import pandas as pd
import matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        return np.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)

    def __repr__(self):
        return f'({self.x},{self.y})'


class Fitness:
    def __init__(self, route):
        self.route = route
        self._distance = 0
        self._fitness = 0

    @property
    def distance(self):
        if self._distance == 0:
            path_distance = 0
            for i in range(len(self.route)):
                from_city = self.route[i]
                to_city = self.route[(i + 1) % len(self.route)]
                path_distance += from_city.distance(to_city)
            self._distance = path_distance
        return self._distance

    @property
    def fitness(self):
        if self._fitness == 0:
            self._fitness = 1 / float(self.distance)
        return self._fitness


def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route


def create_initial_population(population_size, city_list):
    population = []
    for i in range(population_size):
        population.append(create_route(city_list))
    return population


def rank_routes(population):
    fitness_results = {}
    for i in range(len(population)):
        fitness_results[i] = Fitness(population[i]).fitness
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


def selection(population_ranked, elite_size):
    selection_results = []
    df = pd.DataFrame(np.array(population_ranked), columns=['Index', 'Fitness'])
    df['cum_sum'] = df['Fitness'].cumsum()
    df['cum_perc'] = 100 * df['cum_sum'] / df['Fitness'].sum()

    for i in range(elite_size):
        selection_results.append(population_ranked[i][0])

    for i in range(len(population_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(len(population_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(population_ranked[i][0])
                break

    return selection_results


def mating_pool(population, selection_results):
    pool = []
    for i in range(len(selection_results)):
        idx = selection_results[i]
        pool.append(population[idx])
    return pool


def breed(parent1, parent2):

    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent2))

    start_gene = min(gene_a, gene_b)
    end_gene = max(gene_a, gene_b)

    child_p1 = []
    for i in range(start_gene, end_gene):
        child_p1.append(parent1[i])

    child_p2 = [item for item in parent2 if item not in child_p1]
    child = child_p1 + child_p2
    return child


def breed_population(mating_pool, elite_size):
    children = []
    length = len(mating_pool) - elite_size
    pool = random.sample(mating_pool, len(mating_pool))

    for i in range(elite_size):
        children.append(mating_pool[i])

    for i in range(length):
        child = breed(pool[i], pool[-i-1])
        children.append(child)

    return children


def mutate(individual, mutation_rate):
    # This has mutation_rate chance of swapping ANY city, instead of having mutation_rate chance of doing
    # a swap on this given route...
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]

    return individual


def mutate_population(population, mutation_rate):
    mutated_pop = []

    for ind in population:
        mutated = mutate(ind, mutation_rate)
        mutated_pop.append(mutated)
    return mutated_pop


def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    mate_pool = mating_pool(current_gen, selection_results)
    children = breed_population(mate_pool, elite_size)
    next_gen = mutate_population(children, mutation_rate)

    return next_gen


def plot_route(route, title=None):
    for i in range(len(route)):
        city = route[i]
        next_city = route[(i+1) % len(route)]
        plt.scatter(city.x, city.y, c='red')
        plt.plot((city.x, next_city.x), (city.y, next_city.y), c='black')
        if title:
            plt.title(title)

    plt.show()


def solve(cities, population_size, elite_size, mutation_rate, generations):
    pop = create_initial_population(population_size, cities)
    best_route_index, best_distance = rank_routes(pop)[0]
    best_route = pop[best_route_index]
    print(f'Initial distance: {1 / rank_routes(pop)[0][1]}')
    plot_route(best_route, 'Initial')

    for g in range(generations):
        pop = next_generation(pop, elite_size, mutation_rate)

        if (g + 1) % 50 == 0:
            best_route_index, best_distance = rank_routes(pop)[0]
            best_route = pop[best_route_index]
            print(f'[{g+1}/{generations}] Best distance: {1 / best_distance}')
            plot_route(best_route, f'Generation {g+1}')

    print(f'Final distance: {1 / rank_routes(pop)[0][1]}')
    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]
    return best_route


if __name__ == '__main__':
    city_list = []
    for i in range(0, 25):
        city_list.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

    best_route = solve(city_list, 100, 20, 0.01, 500)
    plot_route(best_route, 'Solution')
