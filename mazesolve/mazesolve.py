import cv2
import numpy as np
from py_star import a_star
from time import time
from os.path import basename, join, splitext, abspath, dirname

def main():
    #maze_name = input("Please enter the name of the maze picture with ending: ")
    maze_name = "Sketch.png"
    #diagonal_ok = input("Is traveling diagonally allowed? : ")
    diagonal_ok = False
    if diagonal_ok == "yes":
        diagonal_ok = True
    elif diagonal_ok == "no":
        diagonal_ok = False
    else:
        print("Invalid input! Write 'yes' or 'no'.")
        print("Assuming diagonal travel is not allowed.")
        diagonal_ok = False

    files = parse_filepath(maze_name)
    solve_maze(files[0], files[1], diagonal_ok)


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
    grid[grid == 0] = -1#np.iinfo(np.int32).max
    grid[grid == 255] = 1

    # start is the first white block in the top row
    start_j, = np.where(grid[0, :] == 1)
    start = np.array([0, start_j[0]])

    # end is the first white block in the final column
    end_i, = np.where(grid[:, -1] == 1)
    end = np.array([end_i[0], grid.shape[1] - 1])
    t0 = time()
    # set diagonal_ok=True to enable 8-connectivity
    path, tried_pixels = a_star(grid, start, end, diagonal_ok)
    dur = time() - t0
    #path is tuple of two arrays containing x and y values of path nodes
    if path != None:
        print("found path of length {0} and expanded {1} nodes in {2}s".format(len(path[0]),len(tried_pixels[0]), dur))
        maze[tried_pixels] = (0,255,255)
        maze[path] = (0, 0, 255)
        print("plotting path to {0}".format(solution_path))
        cv2.imwrite(solution_path, maze)
    else:
        print("no path found")
        maze[tried_pixels] = (0,255,255)
        print("plotting expanded pixels to {0}".format(solution_path))
        cv2.imwrite(solution_path, maze)

    print("done")

if __name__ == "__main__":
    main()