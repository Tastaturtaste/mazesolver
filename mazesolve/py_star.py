import numpy as np

import inspect
from os.path import abspath, dirname, join
import a_star_algo

def a_star(weights, start, goal, diagonal_ok):
    # every weight has to be at least one
    #if weights.min(axis=None) < 1:
    #    raise ValueError("Minimum cost to move must be 1, but got {0}".format(weights.min(axis=None)))
    
    #Ensure start is within grid
    if(start[0] < 0 or start[1] >= weights.shape[0] or start[1] < 0 or start[1] >= weights.shape[1]):
        raise ValueError("Start of ({0}) lies outside grid.".format(start))
    
    #Ensure goal is within grid
    if(goal[0] < 0 or goal[1] >= weights.shape[0] or goal[1] < 0 or goal[1] >= weights.shape[1]):
        raise ValueError("Goal of ({0}) lies outside grid.".format(goal))
    
    height, width = weights.shape
    start_index = int(np.ravel_multi_index(start, (height, width)))
    goal_index = int(np.ravel_multi_index(goal, (height, width)))
    weights = weights.flatten()
    weights[start_index] = 0.0;
    weights[goal_index] = 0.0;
    path, tried_pixels = invoke_pathfinder(width,height,weights,start_index,goal_index,diagonal_ok)
    return (path,tried_pixels)

def invoke_pathfinder(width,height,weights,start_index,goal_index,diagonal_ok):
    #Invoking C++ stuff here
    path, tried_pixels = a_star_algo.get_path(width, height, weights, start_index, goal_index, diagonal_ok)
    if path[0] >= 0:
        return (np.unravel_index(path, (height,width)),np.unravel_index(tried_pixels,(height,width)))
    else:
        return (None,np.unravel_index(tried_pixels,(height,width)))
    #If no path was found returns an empty tuple, otherwise tuple of path indices
