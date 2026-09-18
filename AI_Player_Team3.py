from typing import Dict, List, Optional, Tuple, Callable
from halma import *

# TO DO
# Implement maxn search to a depth of 2 (looking 2 moves ahead for your team).
# Implement pruning of repeated positions in the tree.
# Implement branch ordering using heuristics.
# Implement Alpha-Beta pruning to avoid searching hopeless branches.
# Optimise your code so that it runs efficiently.
# When the binary flag is set the code should visualize the search tree using treelib
# and output the result to Team<X>_Tree.png

def AI_Player_Team3(
        board: List[List[int]],
        player: int,
        visualize_tree: bool
) -> Tuple[str, str]:

# CHECK IF PLAYER IS VALID
    if player not in [1, 2, 3, 4]:
            raise ValueError(f"Player {player} is not a valid player")

# CHECK IF BOARD IS 5x5
    if len(board) != 5 or any(len(row) != 5 for row in board):
            raise ValueError("Board must be 5 by 5")
    
# INITIALIZE EMPTY LIST FOR LEGAL MOVES
    legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    
# SEARCH ENTIRE BOARD ADDING ALL LEGAL MOVES
    for row in range(5):
            for column in range(5):
                if board[row][column] != player:
                    continue
    
                oldPos: Tuple[int, int] = (row, column)
    
                for new_row in range(5):
                    for new_column in range(5):
                        newPos: Tuple[int, int] = (new_row, new_column)
    
                        if check_legal_move(board, oldPos, newPos):
                            legal_moves.append((oldPos, newPos))
                            
    if not legal_moves:
            raise ValueError(f"Player {player} has no legal moves")
        
    # SOME LOGIC HERE TO DO:
    
    
    
    # CONVERT COORDINATES TO BOARD NOTATION
    old_reference: str = chr(ord("A") + oldPos[1]) + str(oldPos[0] + 1)
    new_reference: str = chr(ord("A") + newPos[1]) + str(newPos[0] + 1)
    
    return old_reference, new_reference