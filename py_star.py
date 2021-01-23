import numpy as np
import math
from sortedcontainers import SortedSet

class Node:
    def __init__(self, idx: int, sureCost: float, heuristicCost: float, combinedCost: float, parent):
        self.idx = idx # Unique per node
        self.sureCost = sureCost
        self.heuristicCost = heuristicCost
        self.combinedCost = combinedCost
        self.parent = parent

    def __lt__(self, r):
        if self.combinedCost < r.combinedCost:
            return True
        elif self.idx < r.idx:
            return True
        else:
            return False
    
    def __repr__(self):
        return f"Node(idx={self.idx}, sureCost={self.sureCost}, heuristicCost={self.heuristicCost}, combinedCost={self.combinedCost}, parent={id(self.parent)}"

def heuristicCost(pos1: tuple, pos2: tuple, diagonalOk: bool):
    DIST1 = abs(pos1[0] - pos2[0])
    DIST2 = abs(pos1[1] - pos2[1])
    if diagonalOk:
        return math.sqrt(DIST1**2 + DIST2**2)
    else:
        return DIST1 + DIST2

def posFromIndex(index: int, width: int):
    return (index % width, index // width)

def constructPath(start: Node, end: Node):
    # Beginning with the start Node traverses through the Node pointers and adds the indices to the path
	# Expects every Node to point to the following Node in the path
    path = [start.idx]
    current = start
    while current != end:
        parent = current.parent
        path.append(parent.idx)
        current = parent

    return path

def py_star(width, height, costs, startIndex, endIndex, diagonalOk):
    if (width < 0 or height < 0):
        raise ValueError("Width and height have to be positive!")
    if (width * height != len(costs)):
        raise ValueError("Width * height != len(costs)!")
    if (startIndex < 0) or (startIndex >(len(costs) - 1)) or (endIndex < 0) or (endIndex > (len(costs) - 1)):
        raise ValueError(f"Start and end indices have to be in the range [0, {len(costs)})!")

    # find path from exit to start, this way when traversing the nodes from the start
	# every node points to the next one in the path
    startIndex, endIndex = endIndex, startIndex
    startPos = (startIndex % width, startIndex // width)
    endPos = endIndex % width, endIndex / width
    nodeMap = [Node(idx, math.inf, 0.0, math.inf, None) for idx in range(0,len(costs))]
    endNode = nodeMap[endIndex]
    startNode = nodeMap[startIndex]

    startNode.sureCost = 0
    startNode.heuristicCost = heuristicCost(startPos, endPos, diagonalOk)
    startNode.combinedCost = startNode.sureCost + startNode.heuristicCost

    DIAG_COST = math.sqrt(2)
    openlist = SortedSet([startNode],key=lambda node: node.combinedCost)
    closedlist = set()
    while len(openlist) > 0:
        current = openlist.pop(0)
        if current == endNode:
            # call with end and start switched to get correct direction back
            return (constructPath(endNode, startNode), closedlist)
        closedlist.add(current.idx)
        curX, curY = posFromIndex(current.idx, width)
        for dx in range(-1,2):
            for dy in range(-1,2):
                # skip diagonal entrys if diagonals are not viable
                if not diagonalOk and (abs(dx) == abs(dy)):
                    continue
                x, y = curX + dx, curY + dy
                # skip if node would go outside rectangle
                # cannot wrap with unsigned cast like in cpp
                if (x-width+1)*x > 0 or (y - height+1)*y > 0:
                    continue
                neighbor = nodeMap[current.idx + dx + dy*width]
                # skip previously visited nodes, including the current node
                if neighbor.idx in closedlist:
                    continue
                # skip if node is not passable
                if costs[neighbor.idx] < 0:
                    continue
                diagonalMove = (dx*dy) != 0
                newSureCost = current.sureCost + (DIAG_COST if diagonalMove  else 1) * costs[neighbor.idx]
                if newSureCost < neighbor.sureCost:
                    # Make sure to not invalidate the ordered set
                    openlist.discard(neighbor)
                    neighbor.sureCost = newSureCost
                    neighbor.heuristicCost = heuristicCost((x,y), endPos, diagonalOk)
                    # combined cost for ordering of the open set
                    neighbor.combinedCost = neighbor.sureCost + neighbor.heuristicCost
                    neighbor.parent = current
                    openlist.add(neighbor)

    return ([-1], closedlist)