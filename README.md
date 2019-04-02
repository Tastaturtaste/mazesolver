# a_star
This is my attempt at solving mazes with an A*-Algorithm using a combination of cpp and python with pybind and building an executable with pyinstaller.

To solve a maze just use the "mazesolve.exe" and provide the filename of the maze.
Don't specify any path, just be sure the folder with the "mazesolve.exe" also contains the folder "mazes" and "solutions" and the relevant maze is in the "mazes" folder.

You can also use the python version "mazesolve.py"
Before first use of mazesolve.py run the install.bat that is provided or install the pathfinder package manually.
Make sure the "mazesolve.py" and "py_star.py" are in the same folder as the "mazes" and "solutions" folder like with the executable version.

"py_star.py" contains the function "a_star", which expects a 2D-numpyarray of weights, a blocker_cutoff above which weights will be considered unpassable, start and goal coordinates as tuples and a bool-flag which can enable diagonal pathing. 
The function does necessery conversions, calls into the c++-code and returns a tuple of two numpy arrays which contain row- and column indices.
This function can be imported, just make sure to always group "py_star.py" and "mazesolve.py" together.


If an import error gets triggered while using the script or executable, make sure you have the 
"vc++ redistributable for visual studio 2015" installed. 
Most people should have it already installed. If not or you are unsure you can 
download it here: https://www.microsoft.com/en-us/download/details.aspx?id=48145