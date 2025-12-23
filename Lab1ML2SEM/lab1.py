import numpy as np
import matplotlib.pyplot as plt
import random

n_factories = 5
k_cities = 5
population_size = 50
generations = 500
mutation_rate = 0.5

np.random.seed(42)

factory_capacity = np.random.randint(50, 200, size=n_factories)
city_demand = np.random.randint(20, 100, size=k_cities)
distances = np.random.randint(5, 50, size=(n_factories, k_cities))

def fitness(chromosome):
    city_supplied = chromosome.sum(axis=0)
    penalty_demand = np.sum(np.maximum(city_demand - city_supplied, 0))  # недобор
    excess_supply = np.sum(np.maximum(city_supplied - city_demand, 0))  # избыток
    factory_overload = np.sum(np.maximum(chromosome.sum(axis=1) - factory_capacity, 0))
    transport_cost = np.sum(chromosome * distances)
    return transport_cost + 50 * penalty_demand + 10 * excess_supply + 50 * factory_overload

def create_population():
    pop = []
    for _ in range(population_size):
        chromosome = np.random.randint(1, 20, size=(n_factories, k_cities))
        normalize_chromosome(chromosome)
        pop.append(chromosome)
    return pop



def normalize_chromosome(chromosome):
    for i in range(n_factories):
        total = chromosome[i].sum()
        if total > factory_capacity[i]:
            chromosome[i] = np.floor(chromosome[i] * factory_capacity[i] / total)

    for j in range(k_cities):
        total = chromosome[:, j].sum()
        if total > city_demand[j] * 1.2:
            chromosome[:, j] = np.floor(chromosome[:, j] * city_demand[j] * 1.2 / total)



def one_point_crossover(p1, p2):
    point = np.random.randint(1, n_factories)
    c1 = np.vstack((p1[:point], p2[point:]))
    c2 = np.vstack((p2[:point], p1[point:]))
    normalize_chromosome(c1)
    normalize_chromosome(c2)
    return c1, c2


def two_point_crossover(p1, p2):
    point1, point2 = sorted(np.random.choice(range(1, n_factories), 2, replace=False))
    c1 = np.vstack((p1[:point1], p2[point1:point2], p1[point2:]))
    c2 = np.vstack((p2[:point1], p1[point1:point2], p2[point2:]))
    normalize_chromosome(c1)
    normalize_chromosome(c2)
    return c1, c2


def uniform_crossover(p1, p2):
    mask = np.random.randint(0, 2, size=(n_factories, k_cities))
    c1 = p1 * mask + p2 * (1 - mask)
    c2 = p2 * mask + p1 * (1 - mask)
    normalize_chromosome(c1)
    normalize_chromosome(c2)
    return c1, c2



def mutate_random(chromosome):
    i, j = np.random.randint(0, n_factories), np.random.randint(0, k_cities)
    chromosome[i, j] = max(0, chromosome[i, j] + np.random.randint(-10, 11))
    normalize_chromosome(chromosome)
    return chromosome


def mutate_row_shift(chromosome):
    row = np.random.randint(0, n_factories)
    chromosome[row] = np.roll(chromosome[row], 1)
    normalize_chromosome(chromosome)
    return chromosome


def mutate_col_shift(chromosome):
    col = np.random.randint(0, k_cities)
    chromosome[:, col] = np.roll(chromosome[:, col], 1)
    normalize_chromosome(chromosome)
    return chromosome

def mutate_smart(chromosome):
    city_supplied = chromosome.sum(axis=0)
    for j in range(k_cities):
        if city_supplied[j] < city_demand[j]:
            i = np.random.randint(0, n_factories)
            add_amount = min(city_demand[j] - city_supplied[j], factory_capacity[i] - chromosome[i].sum())
            chromosome[i, j] += add_amount
    normalize_chromosome(chromosome)
    return chromosome


def select(pop):
    sorted_pop = sorted(pop, key=fitness)
    return sorted_pop[:population_size // 2]

population = create_population()
best_fitness_over_time = []

for gen in range(generations):
    selected = select(population)
    offspring = []
    while len(offspring) < population_size:
        parents = random.sample(selected, 2)
        choice = np.random.randint(3)
        if choice == 0:
            c1, c2 = one_point_crossover(parents[0], parents[1])
        elif choice == 1:
            c1, c2 = two_point_crossover(parents[0], parents[1])
        else:
            c1, c2 = uniform_crossover(parents[0], parents[1])
        offspring.extend([c1, c2])

    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            mut_type = np.random.randint(4)
            if mut_type == 0:
                offspring[i] = mutate_random(offspring[i])
            elif mut_type == 1:
                offspring[i] = mutate_row_shift(offspring[i])
            elif mut_type == 2:
                offspring[i] = mutate_col_shift(offspring[i])
            else:
                offspring[i] = mutate_smart(offspring[i])

    population = offspring
    best_fitness_over_time.append(fitness(select(population)[0]))

best_solution = select(population)[0]
print("Лучшее распределение продукции (заводы -> города):")
print(best_solution)
print("Потребности городов:", city_demand)
print("Производственные мощности заводов:", factory_capacity)
print("Fitness лучшего решения:", fitness(best_solution))

plt.plot(best_fitness_over_time)
plt.xlabel("Поколение")
plt.ylabel("Лучший fitness")
plt.title("Генетический алгоритм: оптимизация распределения продукции")
plt.grid(True)
plt.show()
