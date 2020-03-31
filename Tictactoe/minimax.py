# Import the tictactoe class
from tictactoe import Tictactoe
# Import the random number generation library
import random


# Create a subclass of tictactoe with an implementation of
# the agent_move function
class AIGame(Tictactoe):
    # The agent_move function − the function has to tell the
    # tictactoe class what move to make next in the game
    #
    # Inputs: state − current state of the board
    # Returns: new state of the board with the proposed move
    def agent_move(self, state):

        def minimax(position, maximising_player, depth=4):
            if depth == 0 or self.terminal(position):
                return self.evaluate(position)

            if maximising_player:
                leading_value = -1000
                new_states = self.max_possible_moves(position)
                for s in new_states:
                    value = minimax(s, False, depth-1)
                    leading_value = max(leading_value, value)
                return leading_value

            else:
                leading_value = 1000
                new_states = self.min_possible_moves(position)
                for s in new_states:
                    value = minimax(s, True, depth - 1)
                    leading_value = min(leading_value, value)
                return leading_value

        new_states = self.max_possible_moves(state)
        best_value = -1000
        leading_state = new_states[0]
        for s in new_states:
            state_value = minimax(s, False)
            if state_value > best_value:
                best_value = state_value
                leading_state = s
                print(best_value)

        return leading_state


class StupidGame(Tictactoe):
    def agent_move(self, state):
        while True:
            r, c = random.randint(0, 2), random.randint(0, 2)

            if state[r, c] == 0:
                state[r, c] = 1
                return state


if __name__ == "__main__":
    # difficulty = ""
    # while difficulty != "hard" and difficulty != "easy":
    #     difficulty = input("Type either 'hard' or 'easy' to choose difficulty: ").lower()
    #
    # if difficulty == "hard":
    #     AIGame(opponent="x")
    # else:
    #     StupidGame(opponent="x")
    AIGame(opponent="x")
