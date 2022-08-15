from random import uniform, choice
from numpy.random import randint
import numpy as np

TOURNAMENT_SIZE = 20
MIN_LOSS = 0
CHROMOSOME_LENGTH = 8
KNIGHTS_NUMBER = 23

# possible knight moves
MOVES = ((1, 2), (1, -2), (-1, 2), (-1, -2),
         (2, 1), (2, -1), (-2, 1), (-2, -1))

SQUARES = [(x//CHROMOSOME_LENGTH, x % CHROMOSOME_LENGTH) for x in range(CHROMOSOME_LENGTH**2)]


def add_tuples(t1, t2):
    return t1[0] + t2[0], t1[1] + t2[1]


class Individual:

    def __init__(self, empty=False):
        self.genes = []
        if empty:
            return

        while True:
            self.genes.append(self.get_free_square())
            if len(self.genes) == KNIGHTS_NUMBER:
                break

    def get_loss(self):

        loss = 0

        for i in range(len(self.genes)):
            for move in MOVES:

                if add_tuples(self.genes[i], move) in self.genes:
                    loss += 1

        return loss

    def __repr__(self):
        chess_board = np.full((CHROMOSOME_LENGTH, CHROMOSOME_LENGTH), ".")

        for i in range(KNIGHTS_NUMBER):
            chess_board[self.genes[i][0], self.genes[i][1]] = "K"

        return '\n'.join(map(str, chess_board))

    def get_free_square(self):
        return choice(list(set(SQUARES) - set(self.genes)))


class Population:

    def __init__(self, population_size):
        self.population_size = population_size
        self.individuals = [Individual() for _ in range(population_size)]

    def get_lowest_loss(self):

        lowest_loss = self.individuals[0]

        for individual in self.individuals[1:]:
            if individual.get_loss() < lowest_loss.get_loss():
                lowest_loss = individual
        return lowest_loss

    def get_list_with_lowest_loss(self, n):
        self.individuals.sort(key=lambda x: x.get_loss())
        return self.individuals[:n]

    def get_size(self):
        return self.population_size

    def get_individual(self, index):
        return self.individuals[index]

    def save_individual(self, index, individual):
        self.individuals[index] = individual


class GeneticAlgorithm:

    def __init__(self, population_size=100, crossover_rate=0.85, mutation_rate=0.05, elitism_param=10):
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_param = elitism_param

    def run(self):

        pop = Population(self.population_size)

        generation_counter = 0

        while pop.get_lowest_loss().get_loss() != MIN_LOSS:
            generation_counter += 1
            print(f'Generation #{generation_counter} - the best individual is:\n'
                  f'{pop.get_lowest_loss()}\n '
                  f'with loss: {pop.get_lowest_loss().get_loss()}')
            pop = self.evolve_population(pop)

        print(f"The solution of putting {KNIGHTS_NUMBER} knights without collision is:")
        print(pop.get_lowest_loss())

    def evolve_population(self, population):
        next_population = Population(self.population_size-self.elitism_param)
        best_individuals = population.get_list_with_lowest_loss(self.elitism_param)

        next_population.individuals = next_population.individuals + best_individuals

        for index in range(next_population.get_size()):
            if uniform(0, 1) < self.crossover_rate:
                self.create_next_individual(index, population, next_population)

        for individual in next_population.individuals:
            self.mutate(individual)

        return next_population

    def mutate(self, individual):
        for index in range(KNIGHTS_NUMBER):
            if uniform(0, 1) < self.mutation_rate:
                individual.genes[index] = individual.get_free_square()
        return

    def create_next_individual(self, index, population, next_population):
        first = self.random_selection(population)
        second = self.random_selection(population)
        next_population.save_individual(index, self.crossover(first, second))
        return

    @staticmethod
    def crossover(individual1, individual2):
        cross_individual = Individual(empty=True)
        split_point = randint(KNIGHTS_NUMBER)

        for index in range(split_point):
            gen = individual1.genes[index]
            cross_individual.genes.append(gen)

        for index in range(split_point, KNIGHTS_NUMBER):
            gen = individual2.genes[index]
            if gen in cross_individual.genes:
                gen = cross_individual.get_free_square()
            cross_individual.genes.append(gen)

        return cross_individual

    @staticmethod
    def random_selection(actual_population):

        new_population = Population(TOURNAMENT_SIZE)

        for i in range(new_population.get_size()):
            random_index = randint(actual_population.get_size())
            new_population.save_individual(i, actual_population.get_individual(random_index))

        return new_population.get_lowest_loss()


if __name__ == "__main__":
    algorithm = GeneticAlgorithm()
    algorithm.run()
