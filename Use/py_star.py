import numpy as np

import inspect
from os.path import abspath, dirname, join
from pathfinder import get_path, IntVec

def a_star(weights, blocker_cutoff, start, goal, diagonal_ok = False):
    # every weight has to be at least one
    if weights.min(axis=None) < 1:
        raise ValueError("Minimum cost to move must be 1, but got {0}".format(weights.min(axis=None)))

    #Ensure start is within grid
    if(start[0] < 0 or start[1] >= weights.shape[0] or start[1] < 0 or start[1] >= weights.shape[1]):
        raise ValueError("Start of ({0}) lies outside grid.".format(start))
    
    #Ensure goal is within grid
    if(goal[0] < 0 or goal[1] >= weights.shape[0] or goal[1] < 0 or goal[1] >= weights.shape[1]):
        raise ValueError("Goal of ({0}) lies outside grid.".format(goal))

    height, width = weights.shape
    start_index = int(np.ravel_multi_index(start, (height, width)))
    goal_index = int(np.ravel_multi_index(goal, (height, width)))
    cppweights = IntVec(weights.flatten().tolist())

    #Invoking C++ stuff here
    path = get_path(height, width, cppweights, blocker_cutoff, start_index, goal_index, diagonal_ok)
    path = np.unravel_index(path, (height,width))
    #If no path was found returns an empty tuple, otherwise tuple of path indices
    return path