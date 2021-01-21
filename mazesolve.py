import cv2
import numpy as np
import argparse as ap
from py_star import a_star
from time import time
import os

def load_maze(maze_path):
    # loads maze with [row, column, channel]
    maze = cv2.imread(maze_path,cv2.IMREAD_COLOR)
    if maze is None:
        raise ValueError(f"No file found with given path: {maze_path}")
    else:
        print("loaded maze of shape {0}".format(maze.shape[0:2],))
    return maze
    
def solve_maze(maze, diagonal_ok :int =True, start_pos=None, end_pos=None, unpassable_cutoff :int =0, start_threshold :int =128, end_threshold :int =128, pix2cost=lambda x: 255 - x):
    # the first red pixel where the red channel >= start_threshold and every other channel < start_threshold is considered the start of the maze
    if start_threshold > 255 or end_threshold > 255 or start_threshold < 1 or end_threshold < 1:
        raise ValueError("start and end thresholds have to be in range [1,255]!")
    if unpassable_cutoff not in range(0,255):
        raise ValueError(f"unpassable_cutoff has to be in range [0,255] but is {unpassable_cutoff}")
    if not start_pos:
        possible_starts = np.nonzero((maze[:,:,2] >= start_threshold) & (maze[:,:,0] < start_threshold) & (maze[:,:,1] < start_threshold))
        if len(possible_starts[0]) < 1:
            raise ValueError("No admissible start pixel found!")
        start_pos = (possible_starts[0][0], possible_starts[1][0]) # (row, column)
    elif start_pos[0] not in range(0,len(maze)) or start_pos[1] not in range(0,len(maze[0])):
        raise ValueError(f"Provided start position {start_pos} not in area [(0, 0),({len(maze)}, {len(maze[0])})!")
    if not end_pos:
        possible_ends = np.nonzero((maze[:,:,0] >= end_threshold) & (maze[:,:,1] < end_threshold) & (maze[:,:,2] < end_threshold))
        if len(possible_ends[0]) < 1:
            raise ValueError("No admissible end pixel found!")
        end_pos = (possible_ends[0][0], possible_ends[1][0]) # (row, column)
    elif end_pos[0] not in range(0,len(maze)) or end_pos[1] not in range(0,len(maze[0])):
        raise ValueError(f"Provided start position {end_pos} not in area [(0, 0),({len(maze)}, {len(maze[0])})!")
         

    grid = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY).astype(np.int)
    mask = grid <= unpassable_cutoff
    grid[mask] = -1 # darker pixels are unpassable
    grid[~mask] = pix2cost(grid[~mask]) + 1 # map pixel gray value to cost, cost > 1 for heuristic in a_star admissable
    if np.any(grid[~mask] < 1):
        raise ValueError("provided function for pix2cost has to map all values > unpassable_cutoff to positive values!")

    t0 = time()
    # set diagonal_ok=True to enable 8-connectivity
    path, tried_pixels = a_star(grid, start_pos, end_pos, diagonal_ok)
    dur = time() - t0
    return path, tried_pixels, dur
    #path is tuple of two arrays containing x and y values of path nodes
   

if __name__ == "__main__":
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("maze_path",type=str,help="Path to the maze.")
    parser.add_argument("-d","--disable-diagonal",action='store_false', help="Flag to disallow diagonal travel.")
    parser.add_argument("-suf","--solution-suffix",default="solution",help="Suffix appended to the file name of the solution.")
    parser.add_argument("-r","--result-path",help="The directory to save the result to. Defaults to the directory of the input maze.")
    parser.add_argument("-show","--show-solution",action='store_true',help="Open window showing the solution at the end.")
    parser.add_argument("-p2c","--pix2cost",default="255-x",help="Function with single parameter x that maps the grayvalue of a pixel to a pathing cost. Requirement: f(x) >= 0 for all x in the range (unpassable_cutoff, 255]. Bigger values for x indicate a lighter color.")
    parser.add_argument("-s","--start-threshold",default=128,help="Influences which pixel is choosen as the start. The start pixel has to have a red value >= start-threshold and all other channels have to have to be < start-threshold.")
    parser.add_argument("-e","--end-threshold",default=128,help="Influences which pixel is choosen as the end. The end pixel has to have a blue value >= end-threshold and all other channels have to have to be < end-threshold.")
    args = parser.parse_args()
    maze = load_maze(args.maze_path)

    path_list = args.maze_path.split('/')
    file_dir = "/".join(path_list[:-1])
    file_name = path_list[-1].split('.')[0]
    file_ending = path_list[-1].split('.')[-1]

    solution_name = file_name + '_' + args.solution_suffix + "." + file_ending
    if args.result_path:
        result_dir = "/".join(args.result_path.split('/')[:-1])
    else:
        result_dir = file_dir
    solution_path = "/".join([result_dir,solution_name])
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        
    path, tried_pixels, dur = solve_maze(maze, diagonal_ok=args.disable_diagonal, start_threshold=args.start_threshold, end_threshold=args.end_threshold, pix2cost=lambda x: eval(args.pix2cost))
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
    if args.show_solution:
        cv2.imshow(f"Solution for {'.'.join([file_name,file_ending])}",maze)
        cv2.waitKey(0)
    print("done")