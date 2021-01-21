# a_star
This is my attempt at solving mazes with an A*-Algorithm using a combination of c++ and python.

To solve a maze use the python script ```mazesolve.py```.
Before first use of mazesolve.py run ```pip install -r requirements.txt```. 

```py_star.py``` contains the function ```a_star```, which expects a 2D-numpyarray of weights, start and goal coordinates as tuples and a bool-flag which can enable diagonal pathing. 
The function does necessery conversions to call the c++ function, calls into the c++-code and returns a tuple of two numpy arrays which contain row- and column indices. 
```mazesolve.py``` accepts commandline arguments get the path for the maze it should solve. The maze is expected to be a .png-file. The picture is read in as a grey-scale. Grey values of 0 are considered unpassable, values of 255 are considered passable. At the moment no other values are supported.

If an import error gets triggered while using the script or executable, make sure you have the 
correct ```vc++ redistributable``` installed. Most people should have it already installed. If not or you are unsure you can download it here: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads