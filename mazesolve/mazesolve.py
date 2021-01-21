import cv2
import numpy as np
import argparse as ap
from py_star import a_star
from time import time
import os

def load_maze(maze_path):
    maze = cv2.imread(maze_path)
    if maze is None:
        print("no file found: {0}".format(maze_path))
        return
    else:
        print("loaded maze of shape {0}".format(maze.shape[0:2],))
    return maze
    
def solve_maze(maze, diagonal_ok = True):

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
    return path, tried_pixels, dur
    #path is tuple of two arrays containing x and y values of path nodes
   

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("maze_path",type=str)
    parser.add_argument("--disable-diagonal",action='store_false')
    parser.add_argument("--solution-name")
    args = parser.parse_args()
    maze = load_maze(args.maze_path)

    path_list = args.maze_path.split('/')
    file_path = "/".join(path_list[:-1])
    file_name = path_list[-1].split('.')[0]
    file_ending = path_list[-1].split('.')[-1]

    if args.solution_name:
        solution_path = args.solution_name
        if solution_path.count('.') == 0:
            solution_path = solution_path + '.' + file_ending
    else:
        solution_name = file_name + "_solution" + "." + file_ending
        solution_path = "/".join([file_path,solution_name])
    
    solution_dir = "/".join(solution_path.split('/')[:-1])
    if not os.path.isdir(solution_dir):
        os.mkdir(solution_dir)
        
    path, tried_pixels, dur = solve_maze(maze, args.disable_diagonal)
    if path != None:
        print(f"found path of length {len(path[0])} and expanded {len(tried_pixels[0])} nodes in {dur}s")
        maze[tried_pixels] = (0,255,255)
        maze[path] = (0, 0, 255)
        print(f"plotting path to {solution_path}")
        cv2.imwrite(solution_path, maze)
    else:
        print("no path found")
        maze[tried_pixels] = (0,255,255)
        print("plotting expanded pixels to {0}".format(solution_path))
        cv2.imwrite(solution_path, maze)

    print("done")