# a_star
This is my attempt at solving mazes with an A*-Algorithm using a combination of cpp and python with swig.

To solve a maze just use the "mazesolve.py" and provide the filename of the maze.
Don't specify any path, just be sure the folder with the mazesolve.py also contains the folder "mazes" and the relevant maze is in this folder.

"py_star.py" contains only the function "a_star", which expects a 2D-numpyarray of weights, a blocker_cutoff above which weights will be considered unpassable, start and goal coordinates as tuples and a bool-flag which can enable diagonal pathing. 
The function does necessery conversions, calls into the c++-code and returns a tuple of two numpy arrays which contain row- and column indices.
This function can be imported, just make sure to always group "py_star.py", "pathfinder.py" and "_pathfinder.pyd" together.

"pathfinder.py" provides two functions, "get_path" and "parse_string_to_weights" and a template-specification of std::vector<int> named "IntVec" usable in python as returnvalue or argument of named functions. 
"parse_string_to_weights" expects a string containing only 'W', '.' and optionally '\n' and returns an std::vector<int> of weights.
"get_path" expects the dimensions of the array to solve, an std::vector<int> of weights, a blocker_cutoff, start and goal indices and a bool specifying if diagonal_pathfinding is allowed.
"IntVec" is castable to pythonintrinsic- and numpycontainers.
