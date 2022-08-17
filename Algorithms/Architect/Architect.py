import numpy as np
from random import random, choice, choices
from itertools import product as cross_product
from copy import deepcopy


class House:
    def __init__(self, house_row, house_col, container_row, container_col):
        self.house_position = house_row, house_col
        self.container_position = container_row, container_col


class SingleSolution:

    def __init__(self, table, col_containers_num, row_containers_num, houses=[]):
        self.table = table
        self.houses = houses
        self.col_containers_num = col_containers_num
        self.row_containers_num = row_containers_num
        self.rows_num = len(self.table)
        self.cols_num = len(self.table[0])

    def generate_solution(self):
        for row_index in range(0, self.rows_num):
            for col_index in range(0, self.cols_num):
                square = self.table[row_index][col_index]
                if square == 1:
                    neighbors = self.get_available_neighboring_squares(row_index, col_index)
                    container_row, container_col = choice(neighbors)
                    self.table[container_row][container_col] = 2
                    self.houses.append(House(row_index, col_index, container_row, container_col))
        return

    def get_available_neighboring_squares(self, row_index, col_index):
        return self.check_neighboring_squares(row_index, col_index, 0, False)

    def check_neighboring_squares(self, row_index, col_index, condition, diagonal_neighbors=True):
        squares = []
        for i, j in cross_product([-1, 0, 1], [-1, 0, 1]):
            x, y = row_index + i, col_index + j
            if 0 <= x < self.rows_num and 0 <= y < self.cols_num:
                if not diagonal_neighbors:
                    if abs(i) + abs(j) == 2:
                        continue
                if abs(i) + abs(j) == 0:    # omit the initial square
                    continue
                if self.table[x][y] == condition:
                    squares.append((x, y))

        return squares

    def mutate(self, k=1):
        for index in choices(list(range(len(self.houses))), k=k):
            house_row, house_col = self.houses[index].house_position
            container_row, container_col = self.houses[index].container_position
            available_fields = self.get_available_neighboring_squares(house_row, house_col)
            if len(available_fields):
                new_container_row, new_container_col = choice(available_fields)
                self.table[container_row][container_col] = 0
                self.table[new_container_row][new_container_col] = 2
                self.houses[index].container_position = new_container_row, new_container_col

    def fitness(self):

        penalty = 0

        for row_index in range(0, self.rows_num):
            for col_index in range(0, self.cols_num):
                square = self.table[row_index][col_index]
                if square == 2:
                    neighboring_containers = self.check_neighboring_squares(row_index, col_index, 2)
                    penalty += len(neighboring_containers)

        for row_index in range(0, self.rows_num):
            penalty += abs(self.table[row_index].count(2) - self.row_containers_num[row_index])

        transposed_table = np.array(self.table).T.tolist()

        for col_index in range(0, self.rows_num):
            penalty += abs(transposed_table[col_index].count(2) - self.col_containers_num[col_index])

        return penalty

    def __str__(self):
        table = deepcopy(self.table)
        for row_index in range(0, self.rows_num):
            for col_index in range(0, self.cols_num):
                if table[row_index][col_index] == 0:
                    table[row_index][col_index] = "."
                if table[row_index][col_index] == 1:
                    table[row_index][col_index] = "A"
                if table[row_index][col_index] == 2:
                    table[row_index][col_index] = "o"

        table_with_params = [[" "] + self.col_containers_num]
        for row_index in range(0, self.rows_num):
            table_with_params.append([self.row_containers_num[row_index]]+table[row_index])

        return '\n'.join(['\t'.join([str(cell) for cell in row]) for row in table_with_params])


class SimulatedAnnealing:
    def __init__(self, table, col_containers_num, row_containers_num, min_temp, max_temp, cooling_rate=0.999):
        self.table = table
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.col_containers_num = col_containers_num
        self.row_containers_num = row_containers_num
        self.cooling_rate = cooling_rate
        self.actual_state = SingleSolution(table, containers_in_row, containers_in_columns)
        self.best_state = self.actual_state
        self.next_state = None

    def run(self):

        self.actual_state.generate_solution()
        temp = self.max_temp
        counter = 0

        while temp > self.min_temp and self.best_state.fitness() != 0:
            counter += 1

            if counter % 100 == 0:
                print('Iteration #%s - loss: %s' % (counter, self.best_state.fitness()))
                print("")
                print(self.actual_state)

            new_state = self.generate_next_state(self.actual_state)

            actual_energy = self.actual_state.fitness()
            new_energy = new_state.fitness()

            if random() < self.accept_prob(actual_energy, new_energy, temp):
                self.actual_state = new_state

            if self.actual_state.fitness() < self.best_state.fitness():
                self.best_state = self.actual_state

            temp = temp * self.cooling_rate

        print('Solution: \n%s' % self.best_state)

    def generate_next_state(self, actual_state):
        new_state = SingleSolution(actual_state.table,
                                   self.col_containers_num,
                                   self.row_containers_num,
                                   actual_state.houses)
        new_state.mutate()
        return new_state

    @staticmethod
    def accept_prob(actual_energy, next_energy, temp):

        if next_energy < actual_energy:
            return 1

        return np.exp((actual_energy - next_energy) / temp)


if __name__ == '__main__':

    architect_table = [
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 1]
                      ]

    containers_in_columns = [1, 1, 2, 1, 1, 1]
    containers_in_row = [1, 0, 2, 1, 2, 1]

    algorithm = SimulatedAnnealing(architect_table,
                                   containers_in_columns,
                                   containers_in_row,
                                   min_temp=1e-5,
                                   max_temp=1e9)
    algorithm.run()
