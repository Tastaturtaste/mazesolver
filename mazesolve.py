import cv2
import numpy as np
from py_star import a_star
from time import time
from os.path import basename, join, splitext, abspath, dirname

def main():
    maze_name = input("Please enter the name of the maze picture with ending: ")
    files = parse_filepath(maze_name)
    solve_maze(files[0], files[1])


def parse_filepath(maze_name):
    maze_fpath = join(abspath("mazes"), maze_name)
    solution_fpath = join(abspath("Solutions"), "{0}_solution.png".format(splitext(basename(maze_fpath))[0]))
    return [maze_fpath, solution_fpath]
    
def solve_maze(maze_path, solution_path, diagonal_ok = False):
    maze = cv2.imread(maze_path)
    if maze is None:
        print("no file found: {0}".format(maze_path))
        return
    else:
        print("loaded maze of shape {0}".format(maze.shape[0:2],))

    grid = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY).astype(np.int)
    grid[grid == 0] = np.iinfo(np.int32).max
    grid[grid == 255] = 1

    # start is the first white block in the top row
    start_j, = np.where(grid[0, :] == 1)
    start = np.array([0, start_j[0]])

    # end is the first white block in the final column
    end_i, = np.where(grid[:, -1] == 1)
    end = np.array([end_i[0], grid.shape[1] - 1])
    t0 = time()
    # set diagonal_ok=True to enable 8-connectivity
    path = a_star(grid,np.iinfo(np.int32).max, start, end, False)
    dur = time() - t0

    if len(path[0]) > 0:
        print("found path of length {0} in {1}s".format(len(path[0]), dur))
        maze[path] = (0, 0, 255)
        print("plotting path to {0}".format(solution_path))
        cv2.imwrite(solution_path, maze)
    else:
        print("no path found")

    print("done")

if __name__ == "__main__":
    main()