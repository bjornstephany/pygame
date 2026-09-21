from typing import Dict, List, Optional, Tuple, Callable
import subprocess
from halma import *
from treelib import Tree

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
TEAM_NUMBER = 3

# "Depth 2" = we get to move twice (now, and again after the 3 opponents replied).
# One full round is 4 plies (one per player), so:
#   ply 1 = us, ply 2-4 = the three opponents, ply 5 = us again, then evaluate.
MY_MOVES_AHEAD = 2
SEARCH_PLIES = 4 * (MY_MOVES_AHEAD - 1) + 1     # = 5

# The full tree has thousands of nodes, too big for a picture,
# so we only DRAW the first few plies.
TREE_DRAW_PLIES = 2

# Every evaluation gives the 4 players "shares" that always add up to this.
# Knowing the total is what makes the (shallow) pruning possible.
SCORE_TOTAL = 100.0

# These are used while searching (module-level so every function can reach them)
tree: Optional[Tree] = None
node_counter = 0


# ---------------------------------------------------------------------------
# SMALL HELPERS
# ---------------------------------------------------------------------------

# FUNCTION ALTERNATING WHICH PLAYER'S TURN IT IS
def get_next_player(current_player):
    if current_player == 4:
        return 1
    return current_player + 1


# Turn (row, column) into a name like "B3"
def to_reference(pos: Tuple[int, int]) -> str:
    return chr(ord("A") + pos[1]) + str(pos[0] + 1)


# All legal moves for a player.
# OPTIMISATION: instead of testing all 25 squares for every piece, we only test
# the 4 one-step squares and the 4 two-step (jump) squares around the piece.
# Returns an empty list if there are no moves (the search then just skips that turn).
def get_legal_moves(
        board: List[List[int]],
        player: int
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    legal_moves: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (0, 2), (0, -2), (2, 0), (-2, 0)]

    for row in range(5):
        for column in range(5):
            if board[row][column] != player:
                continue
            oldPos = (row, column)
            for d_row, d_col in directions:
                new_row = row + d_row
                new_col = column + d_col
                # check_legal_move does not check the edges of the board, so we do
                if not (0 <= new_row < 5 and 0 <= new_col < 5):
                    continue
                newPos = (new_row, new_col)
                if check_legal_move(board, oldPos, newPos):
                    legal_moves.append((oldPos, newPos))

    return legal_moves


# ---------------------------------------------------------------------------
# HEURISTIC (evaluation function)
# ---------------------------------------------------------------------------

# How good is one square for a player's piece? 8 = on a goal square,
# smaller the further (Manhattan distance) it is from the closest goal square.
def square_value(pos: Tuple[int, int], player: int) -> int:
    closest = 8
    for goal in win_cells_all[player]:
        distance = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
        if distance < closest:
            closest = distance
    return 8 - closest


# Total progress of a player = sum of the values of his 3 pieces
def progress(board: List[List[int]], player: int) -> int:
    total = 0
    for row in range(5):
        for column in range(5):
            if board[row][column] == player:
                total += square_value((row, column), player)
    return total


# Gives a list of 4 scores (index 0 = player 1 ... index 3 = player 4).
# Score = the player's share of all progress. The +1 avoids dividing by zero
# and keeps every score above 0. The 4 scores always add up to SCORE_TOTAL.
def evaluate(board: List[List[int]]) -> List[float]:
    progresses = [progress(board, p) + 1 for p in [1, 2, 3, 4]]
    everything = sum(progresses)
    return [SCORE_TOTAL * p / everything for p in progresses]


# BRANCH ORDERING: best-looking moves first (the ones that bring a piece closest
# to its goal). Good moves first means the pruning can cut more.
def order_moves(moves, player):
    def gain(m):
        return square_value(m[1], player) - square_value(m[0], player)
    return sorted(moves, key=gain, reverse=True)


# ---------------------------------------------------------------------------
# THE SEARCH
# ---------------------------------------------------------------------------

# Maxn search with shallow pruning and repeated-position pruning.
#   board       : the board (we change it and undo it again, no copying)
#   player      : whose turn it is at this node
#   plies_left  : how many more moves to look ahead
#   parent_best : the best score the PREVIOUS player has found so far (own score)
#   seen        : dictionary of positions already searched
#   my_id       : id of this node in the drawn tree (None if we don't draw it)
#   ply         : how deep we are (root's children have ply 1)
# Returns (list of 4 scores, best move)
def maxn(board, player, plies_left, parent_best, seen, my_id, ply):
    global node_counter
    # Leaf: use the heuristic
    if plies_left == 0:
        return evaluate(board), None

    # PRUNING OF REPEATED POSITIONS:
    # same board + same player to move + same depth left = same answer.
    key = (tuple(tuple(row) for row in board), player, plies_left)
    if key in seen:
        if my_id is not None:
            tree.get_node(my_id).tag += " [repeat]"
        return seen[key], None

    moves = order_moves(get_legal_moves(board, player), player)

    # No legal moves: this player passes
    if len(moves) == 0:
        return maxn(board, get_next_player(player), plies_left - 1,
                    0, seen, my_id, ply + 1)

    best_scores = None
    best_move = None
    was_cut = False

    for index, (old_pos, new_pos) in enumerate(moves):
        # make the move
        board[new_pos[0]][new_pos[1]] = player
        board[old_pos[0]][old_pos[1]] = 0

        # add a node to the picture (only for the first few plies)
        child_id = None
        if my_id is not None and ply < TREE_DRAW_PLIES:
            node_counter += 1
            child_id = node_counter
            tree.create_node(f"P{player}: {to_reference(old_pos)}->{to_reference(new_pos)}",
                             child_id, parent=my_id)

        # search deeper. The child needs to know OUR best score so far.
        my_best_so_far = 0
        if best_scores is not None:
            my_best_so_far = best_scores[player - 1]
        scores, _ = maxn(board, get_next_player(player), plies_left - 1,
                         my_best_so_far, seen, child_id, ply + 1)

        # undo the move
        board[old_pos[0]][old_pos[1]] = player
        board[new_pos[0]][new_pos[1]] = 0

        if child_id is not None:
            tree.get_node(child_id).tag += " -> " + format_scores(scores)

        # Maxn: keep the child that is best for the player to move
        if best_scores is None or scores[player - 1] > best_scores[player - 1]:
            best_scores = scores
            best_move = (old_pos, new_pos)

        # ALPHA-BETA (shallow pruning, Korf 1991):
        # All scores add up to SCORE_TOTAL. If we already found a move worth X to us,
        # then the parent player can get at most SCORE_TOTAL - X from this node.
        # If that is no better than what the parent already has, he will never
        # choose this node, so we stop looking at our other moves.
        if best_scores[player - 1] >= SCORE_TOTAL - parent_best - 1e-9:
            was_cut = True
            if my_id is not None:
                skipped = len(moves) - index - 1
                if skipped > 0:
                    tree.get_node(my_id).tag += f" [pruned {skipped} moves]"
            break

    # Only remember complete results (a cut result is only valid for its parent)
    if not was_cut:
        seen[key] = best_scores

    return best_scores, best_move


def format_scores(scores) -> str:
    return "(" + ", ".join(f"{s:.1f}" for s in scores) + ")"


# ---------------------------------------------------------------------------
# THE FUNCTION THE GUI CALLS
# ---------------------------------------------------------------------------
def AI_Player_Team3(
        board: List[List[int]],
        player: int,
        visualize_tree: bool
) -> Tuple[str, str]:
    global tree, node_counter

    # CHECK IF PLAYER IS VALID
    if player not in [1, 2, 3, 4]:
        raise ValueError(f"Player {player} is not a valid player")

    # CHECK IF BOARD IS 5x5
    if len(board) != 5 or any(len(row) != 5 for row in board):
        raise ValueError("Board must be 5 by 5")

    if len(get_legal_moves(board, player)) == 0:
        raise ValueError(f"Player {player} has no legal moves")

    # Set up the tree for drawing (if asked)
    node_counter = 0
    if visualize_tree:
        tree = Tree()
        tree.create_node(f"P{player} to move (root)", 0)
        root_id = 0
    else:
        tree = None
        root_id = None

    # work on a copy so the real board can never be damaged
    work_board = [row[:] for row in board]
    scores, best_move = maxn(work_board, player, SEARCH_PLIES, 0, {}, root_id, 0)

    if visualize_tree:
        save_tree()

    old_pos, new_pos = best_move
    return to_reference(old_pos), to_reference(new_pos)


# Write the tree as a graphviz .dot file and turn it into Team3_Tree.png
def save_tree():
    dot_name = f"Team{TEAM_NUMBER}_Tree.dot"
    png_name = f"Team{TEAM_NUMBER}_Tree.png"
    tree.to_graphviz(filename=dot_name, shape="box")
    try:
        subprocess.run(["dot", "-Grankdir=LR", "-Tpng", dot_name, "-o", png_name], check=True)
    except Exception as error:
        print(f"Could not make the PNG (is Graphviz installed?): {error}")