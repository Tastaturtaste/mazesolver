# a_star
This is my attempt at solving mazes with an A*-Algorithm using a combination of cpp and python with pybind.

Before first use run the install.bat that is provided or install the pathfinder package manually. 

To solve a maze just use the "mazesolve.py" and provide the filename of the maze.
Don't specify any path, just be sure the folder with the mazesolve.py also contains the folder "mazes" and the relevant maze is in this folder.

"py_star.py" contains only the function "a_star", which expects a 2D-numpyarray of weights, a blocker_cutoff above which weights will be considered unpassable, start and goal coordinates as tuples and a bool-flag which can enable diagonal pathing. 
The function does necessery conversions, calls into the c++-code and returns a tuple of two numpy arrays which contain row- and column indices.
This function can be imported, just make sure to always group "py_star.py" and "mazesolve.py" together.