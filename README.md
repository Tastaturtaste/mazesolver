# A*-Mazesolver
This is my attempt at solving mazes with an A*-Algorithm using a combination of c++ and python.



## Setup
### Optional: Setup virtual environment
In the root directory of this project run ```python -m venv venv-name``` where ```venv-name``` can be any name you like.  
Activate it using ```venv-name/scripts/activate```.  
Read more about it [here](https://docs.python.org/3/tutorial/venv.html).

### Required
In the root directory run ```pip install -r requirements.txt```.

## Usage

To solve a maze run ```mazesolve.py path/to/maze.png```.


```mazesolve.py``` reads in the three color channels of the picture. The start position for the maze should be marked in red, the exit in blue. Then the picture is converted to gray-scale. Per default grey values of 0 are considered unpassable, values above are considered passable. Grey values that are passable are translated to pathing costs.  
```mazesolve.py``` calls the function ```a_star```, which expects a 2D-numpyarray of weights, start and goal coordinates as tuples and a bool-flag which can enable diagonal pathing. The function does necessery conversions to call the c++ function, calls into the c++-code and returns a tuple of two numpy arrays which contain row- and column indices. The result is drawn into the maze and saved to a file.  
The behaviour can be influenced via the commandline arguments passed to ```mazesolve.py```.

If an import error gets triggered while using the script or executable, make sure the package is correctly installed and you have the correct ```vc++ redistributable``` on the system. Most people should have it already installed. If not or you are unsure you can download it here: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads