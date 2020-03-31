import time
from eightpuzzle import EightPuzzle

puzzle = EightPuzzle(mode="hard")

init_state = puzzle.reset()
goal_state = puzzle.goal()


class Node:
    def __init__(self, s, parent=None, g=0, h=0, action=None):
        self.s = s
        self.parent = parent
        self.g = g
        self.f = g + h
        self.action = action


# Function computing misplaced tiles
def heuristic(s, goal):
    h = 0
    # Walk through all the tiles in the current state
    for i in range(len(s)):
        if s[i] != goal[i]:
            h += 1
    return h


# Function for computing Manhattan Distance
def manhattan(s):
    h = 0
    for i, tile in enumerate(s):
        state_row, state_col = divmod(i, 3)
        goal_row, goal_col = divmod(tile, 3)
        h += abs(goal_row-state_row) + abs(goal_col-state_col)

    return h


start_time = time.time()
print(start_time)

root_node = Node(s=init_state, parent=None, g=0,
                 h=manhattan(s=init_state))
fringe = [root_node]
visited = set(str(init_state))


solution_node = None
while len(fringe) > 0:
    current_node = fringe.pop(0)
    current_state = current_node.s
    if current_state == goal_state:
        solution_node = current_node
        break
    else:
        available_actions = puzzle.actions(s=current_state)
        for a in available_actions:
            next_state = puzzle.step(s=current_state, a=a)
            if str(next_state) in visited:
                continue
            else:
                new_node = Node(s=next_state, parent=current_node,
                                g=current_node.g+1, h=manhattan(s=next_state),
                                action=a)
                fringe.append(new_node)
                visited.add(str(next_state))
        fringe.sort(key=lambda x: x.f)

action_sequence = []

if solution_node is None:
    print("Did not find a solution.")
else:

    next_node = solution_node
    while True:
        if next_node == root_node:
            break

        action_sequence.append(next_node.action)
        next_node = next_node.parent

    action_sequence.reverse()
    print("Number of moves: {}".format(solution_node.g))


elapsed_time = time.time() - start_time
print("Time to find a solution: {:f}".format(elapsed_time))

puzzle.show(a=action_sequence)
