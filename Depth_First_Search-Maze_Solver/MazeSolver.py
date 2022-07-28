from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np


class MazeSolver:
    def __init__(self, maze):
        # a maze is represented by 2D array
        # 0 is an obstacle, and is 1 a valid cell
        self.maze = maze
        self.maze_width = len(maze[0])  # rectangular mazes are allowed
        self.maze_height = len(maze)
        # a list to store the solution
        self.solution = [[0 for _ in range(self.maze_width)] for _ in range(self.maze_height)]

    def find_solution(self):
        if self.solve(0, 0):
            return self.solution
        else:
            print('The solution does not exist')
            return None

    def solve(self, row, col):
        # the function is recursive

        # if we reached the right bottom corner the maze is solved
        if self.is_finished(row, col):
            return True

        # check if the given cell is an obstacle or not
        if self.is_obstacle(row, col):
            # it is valid so it is part of the solution
            self.solution[row][col] = 1

            # goes right if possible
            if col + 1 < self.maze_width and self.solution[row][col + 1] != 1:
                if self.solve(row, col + 1):
                    return True

            # goes left if possible
            if col - 1 >= 0 and self.solution[row][col - 1] != 1:
                if self.solve(row, col - 1):
                    return True

            # goes downwards if possible
            if row + 1 < self.maze_height and self.solution[row + 1][col] != 1:
                if self.solve(row + 1, col):
                    return True

            # goes upwards if possible
            if row - 1 >= 0 and self.solution[row - 1][col] != 1:
                if self.solve(row - 1, col):
                    return True

            # if no other option is available it goes backward
            self.solution[row][col] = 0

        return False

    def is_obstacle(self, row, col):
        if self.maze[row][col] != 1:
            return False

        return True

    def is_finished(self, row, col):
        if row == self.maze_height - 1 and col == self.maze_width - 1:
            self.solution[row][col] = 1
            return True

        return False

    def plot_maze_and_solution(self):
        fig, ax = plt.subplots(1, 2)
        cmap1 = colors.ListedColormap(['white', 'black'])
        cmap2 = colors.ListedColormap(['white', 'black', "red"])
        ax[0].imshow(self.maze, cmap=cmap1)
        ax[0].axis("off")
        ax[0].set_title("Depth_First_Search-Maze_Solver")

        ax[1].imshow(np.asarray(self.solution) + np.asarray(self.maze), cmap=cmap2)
        ax[1].axis("off")
        ax[1].set_title("Solution")
        plt.show()

        return


if __name__ == '__main__':
    maze = [[1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1]]

    maze_solver = MazeSolver(maze)
    maze_solution = maze_solver.find_solution()

    if maze_solution:
        maze_solver.plot_maze_and_solution()
