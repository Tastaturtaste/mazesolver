import cv2
import numpy as np
from os.path import abspath, join, basename, splitext
from os import getcwd


maze_name = input("Please enter the name of the maze picture with ending: ")
maze_fpath = join(abspath(getcwd()), maze_name)
target_name = "{0}_parsed.txt".format(splitext(basename(maze_fpath))[0])
maze = cv2.imread(maze_fpath)
if maze is None:
    print("no file found: {0}".format(maze_fpath))
    exit()
else:
    print("loaded maze of shape {0}".format(maze.shape[0:2],))

numbers = cv2.cvtColor(maze, cv2.COLOR_BGR2GRAY).astype(np.int)
numbers[numbers == 0] = np.iinfo(np.int32).max
numbers[numbers == 255] = 1

np.savetxt(target_name, numbers.flatten(), fmt="%1.1i", delimiter=",", newline=",")