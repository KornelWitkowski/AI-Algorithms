import pytest
from MazeSolver import MazeSolver

maze_1 = [[1, 1], [0, 1]]
maze_2 = [[1, 1], [0, 1]]
maze_3 = [[1, 1], [0, 0], [1, 1]]   # a maze with a dead end

@pytest.mark.parametrize(
    "maze, true_solution",
    [[maze_1, maze_1],
     [maze_2, maze_2],
     [maze_3, None]])
def test_solution(maze, true_solution):

    maze_solver = MazeSolver(maze)
    solution = maze_solver.find_solution()

    assert solution == true_solution

