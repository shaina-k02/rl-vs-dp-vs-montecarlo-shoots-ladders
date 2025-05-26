# Reinforcement Learning

In this project, we are solving a variant of "Shoots and Ladders" to compare the algorithmic effectiveness of the following methods:
1. Monte Carlo Simulation
2. Temporal Difference Algorithms
3. Dynamic programming

The rules are as follows:

- A player starts on Square 0: just off the board on the lower left corner.   Thus there are 101 squares of the game.  
The goal is to reach Square 100, the goal square, in as few moves as possible.
Players take turns.  The first player to the goal wins.  
- A player does not need to land exactly on the goal to win.  For example, if a player is on Square 99 and moves "3", they will reach the goal.
- Anytime a player lands on the square at the top of a chute, they must slide down to the square at the bottom of the chute to complete that turn.
- Anytime a player lands on the bottom square of a ladder, they must climb up to the square at the top of the ladder to complete that turn.
- Though there are 101 squares in the game, the number of states if fewer.   For example, Square 95 is not a "state" because it is impossible to ever be in that state.  The player would slide down the chute immediately.   Tops of chutes and bottoms of ladders are not states.  
- We will define the reward value to be r=-1 for each turn of the game.  By maximizing the rewards, we will minimize the number of turns to the goal state.  
- The action space is (red, blue, green, black) -- see the dice explanation below.

Because the regular game only offers a spinner (no strategy) we will alter the game by replacing the spinner with a set of four die.    The die are Effron Dice (which are probabilistically intransitive).    Below are the four die choices along with the values on their faces.  

RED:  2, 2, 2, 2, 6, 6

BLUE: 3, 3, 3, 3, 3, 3

GREEN: 1, 1, 1, 5, 5, 5

BLACK: 0, 0, 4, 4, 4, 4

For each move, a player can select one of the four dice and roll that dice.   The value on the face is the number of squares they shall move for that turn.  In this way, there is still a stochastic element (the dice roll), but there is now also a strategic element.  It is likely favorable to select different dice from different states of the game.

###Goal
Compute:
1. Policy: the optimal decision for each state of the game as to which
dice to select.   The policy will map states to dice choices.
2. Value V(s):  this is the value of being in state s -- the sum of future rewards when starting in state s.    This will be the negative value of the average number of moves it will take a player to reach the goal from that state s.  
3. Values Q(s,a):  Same as V(s) except we also specify one of the dice color choices as an action.
