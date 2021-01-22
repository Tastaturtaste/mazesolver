# A*-Mazesolver
Example project using the a_star extension module.

## Setup
### Optional: Setup virtual environment
In the root directory of this project run ```python -m venv venv-name``` where ```venv-name``` can be any name you like.  
Activate it using ```venv-name/scripts/activate```.  
Read more about it [here](https://docs.python.org/3/tutorial/venv.html).

### Required
For this step a compatible c++-compiler is likely necessary, since the c++-binary extension has to interact with the python interpreter. Linux users should have one by default while windows users may have to install one. But first proceed as follows.
In the root directory run ```pip install -r requirements.txt```. This will install all required dependencies as well as my c++-extension module.  
If you did not have the correct compiler installed at this stage and the provided wheel is not suitable for your platform the generated error message will tell you which compiler you need and where you can get one. 

## Usage

To solve a maze run ```mazesolve.py path/to/maze.png```.

```mazesolve.py``` reads in the three color channels of the picture. The start position for the maze should be marked in red, the exit in blue. Then the picture is converted to gray-scale. Per default grey values of 0 are considered unpassable, values above are considered passable. Grey values that are passable are translated to pathing costs.  
```mazesolve.py``` calls the function ```a_star```, which expects a 2D-numpyarray of weights, start and goal coordinates as tuples and a bool-flag which can enable diagonal pathing. The function does necessery conversions to call the c++ function, calls into the c++-code and returns a tuple of two numpy arrays which contain row- and column indices.  
The result is drawn into the maze and saved to a file.  
The behaviour can be influenced via the commandline arguments passed to ```mazesolve.py```.  
These can be explored with ```python mazesolve.py -h```

If an import error gets triggered while using the script, make sure the package is correctly installed and you have the correct ```vc++ redistributable``` on the system. Most people should have it already installed. If not or you are unsure you can download it here: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads