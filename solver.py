import numpy as np


def calculate_fitness(solution, distance_table):
    distance = 0
    for i in range(len(solution)):
        distance += distance_table[solution[i], solution[(i + 1) % len(solution)]]
    return 1 / distance


def initialize_population(population_size, distance_table):
    population = []
    for i in range(population_size):
        s = list(range(distance_table.shape[0]))
        np.random.shuffle(s)
        population.append((calculate_fitness(s, distance_table), s))
    population.sort(key=lambda x: -x[0])

    return population


def evaluate(population, distance_table):
    res = []
    for p in population:
        res.append((calculate_fitness(p, distance_table), p))
    return res


def select(population, n):
    draw = np.random.exponential(size=(n,)) * 10
    idx = np.minimum(draw.astype(int), len(population) - 1)
    return np.array(population)[idx].tolist()


def crossover(parents):
    res = []
    for i in range(0, len(parents) - 1, 2):
        slice = np.random.randint(0, len(parents[0]) - 1)
        child = parents[i][1][:slice] + parents[i + 1][1][slice:]
    return res


def mutate(parents):
    res = []
    for parent in parents:
        child = parent[1][:]
        i1 = np.random.randint(0, len(parent) - 1)
        i2 = np.random.randint(0, len(parent) - 1)
        child[i1], child[i2] = child[i2], child[i1]
        res.append(child)
    return res


def solve(distance_table, population_size=20, mutation_rate=0.1, crossover_rate=0.3, steps=1000):
    population = initialize_population(population_size, distance_table)
    best_solution = population[0]

    n_parents = int(population_size * crossover_rate)
    n_parents -= n_parents % 2
    n_mutate = int(population_size * mutation_rate)
    n_rest = max(0, population_size - n_parents - n_mutate)

    for step in range(steps):
        new_population = []
        crossover_res = crossover(select(population, n_parents))
        mutate_res = mutate(select(population, n_mutate))
        new_population.extend(evaluate(crossover_res + mutate_res, distance_table))
        new_population.extend(select(population, n_rest))
        new_population.sort(key=lambda x: -x[0])

        population = new_population

        if population[0][0] > best_solution[0]:
            best_solution = population[0]

        if (step + 1) % 50 == 0:
            print(f'[{step + 1}/steps] Best score: {best_solution[0]:.4f}  Best solution: {best_solution[1]}')


if __name__ == '__main__':
    distance_table = np.loadtxt('data/five_d.txt')
    solve(distance_table)
    pass
