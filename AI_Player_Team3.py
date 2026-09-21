from typing import List, Optional, Tuple
import subprocess

from halma import *
from treelib import Tree

MY_MOVES_AHEAD = 2
SEARCH_PLIES = 4 * (MY_MOVES_AHEAD - 1) + 1

TREE_DRAW_PLIES = 2
SCORE_TOTAL = 100.0

tree: Optional[Tree] = None
node_counter = 0


def get_next_player(player):
    if player == 4:
        return 1
    return player + 1


def to_reference(position: Tuple[int, int]) -> str:
    row, column = position
    return chr(ord("A") + column) + str(row + 1)


def get_legal_moves(
    board: List[List[int]],
    player: int
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:

    moves = []

    directions = [
        (0, 1), (0, -1),
        (1, 0), (-1, 0),
        (0, 2), (0, -2),
        (2, 0), (-2, 0)
    ]

    for row in range(5):
        for column in range(5):

            if board[row][column] != player:
                continue

            old_position = (row, column)

            for row_change, column_change in directions:
                new_row = row + row_change
                new_column = column + column_change

                if new_row < 0 or new_row >= 5:
                    continue

                if new_column < 0 or new_column >= 5:
                    continue

                new_position = (new_row, new_column)

                if check_legal_move(board, old_position, new_position):
                    moves.append((old_position, new_position))

    return moves


def square_value(position: Tuple[int, int], player: int) -> int:
    best_distance = 8

    for goal in win_cells_all[player]:
        distance = abs(position[0] - goal[0])
        distance += abs(position[1] - goal[1])

        if distance < best_distance:
            best_distance = distance

    return 8 - best_distance


def progress(board: List[List[int]], player: int) -> int:
    total = 0

    for row in range(5):
        for column in range(5):

            if board[row][column] == player:
                total += square_value((row, column), player)

    return total


def evaluate(board: List[List[int]]) -> List[float]:
    scores = []

    for player in [1, 2, 3, 4]:
        scores.append(progress(board, player) + 1)

    total = sum(scores)

    for i in range(4):
        scores[i] = SCORE_TOTAL * scores[i] / total

    return scores


def order_moves(moves, player):
    ordered_moves = []

    for move in moves:
        old_position, new_position = move

        gain = (
            square_value(new_position, player)
            - square_value(old_position, player)
        )

        ordered_moves.append((gain, move))

    ordered_moves.sort(reverse=True)

    return [move for gain, move in ordered_moves]


def format_scores(scores) -> str:
    return "(" + ", ".join(f"{score:.1f}" for score in scores) + ")"


def maxn(
    board,
    player,
    plies_left,
    parent_best,
    seen,
    parent_id,
    ply
):
    global node_counter

    if plies_left == 0:
        return evaluate(board), None

    key = (
        tuple(tuple(row) for row in board),
        player,
        plies_left
    )

    if key in seen:

        if parent_id is not None:
            tree.get_node(parent_id).tag += " [repeat]"

        return seen[key], None

    moves = get_legal_moves(board, player)
    moves = order_moves(moves, player)

    if len(moves) == 0:
        next_player = get_next_player(player)

        return maxn(
            board,
            next_player,
            plies_left - 1,
            0,
            seen,
            parent_id,
            ply + 1
        )

    best_scores = None
    best_move = None
    was_pruned = False

    for index, move in enumerate(moves):

        old_position, new_position = move

        old_row, old_column = old_position
        new_row, new_column = new_position

        board[old_row][old_column] = 0
        board[new_row][new_column] = player

        child_id = None

        if parent_id is not None and ply < TREE_DRAW_PLIES:

            node_counter += 1
            child_id = node_counter

            move_name = (
                f"P{player}: "
                f"{to_reference(old_position)}"
                f"->{to_reference(new_position)}"
            )

            tree.create_node(
                move_name,
                child_id,
                parent=parent_id
            )

        previous_best = 0

        if best_scores is not None:
            previous_best = best_scores[player - 1]

        next_player = get_next_player(player)

        scores, unused_move = maxn(
            board,
            next_player,
            plies_left - 1,
            previous_best,
            seen,
            child_id,
            ply + 1
        )

        board[new_row][new_column] = 0
        board[old_row][old_column] = player

        if child_id is not None:
            tree.get_node(child_id).tag += " -> " + format_scores(scores)

        if (
            best_scores is None
            or scores[player - 1] > best_scores[player - 1]
        ):
            best_scores = scores
            best_move = move

        maximum_possible = SCORE_TOTAL - parent_best

        if best_scores[player - 1] >= maximum_possible - 1e-9:

            was_pruned = True

            if parent_id is not None:
                skipped = len(moves) - index - 1

                if skipped > 0:
                    tree.get_node(parent_id).tag += (
                        f" [pruned {skipped} moves]"
                    )

            break

    if not was_pruned:
        seen[key] = best_scores

    return best_scores, best_move


def AI_Player_Team3(
    board: List[List[int]],
    player: int,
    visualize_tree: bool
) -> Tuple[str, str]:

    global tree, node_counter

    if player not in [1, 2, 3, 4]:
        raise ValueError("Player must be 1, 2, 3, or 4")

    if len(board) != 5 or any(len(row) != 5 for row in board):
        raise ValueError("Board must be 5 by 5")

    legal_moves = get_legal_moves(board, player)

    if len(legal_moves) == 0:
        raise ValueError("Player has no legal moves")

    node_counter = 0

    if visualize_tree:
        tree = Tree()
        tree.create_node(
            f"P{player} to move (root)",
            0
        )
        root_id = 0

    else:
        tree = None
        root_id = None

    work_board = [row[:] for row in board]

    scores, best_move = maxn(
        work_board,
        player,
        SEARCH_PLIES,
        0,
        {},
        root_id,
        0
    )

    if visualize_tree:
        save_tree()

    old_position, new_position = best_move

    return (
        to_reference(old_position),
        to_reference(new_position)
    )


def save_tree():

    dot_name = f"Team3_Tree.dot"
    png_name = f"Team3_Tree.png"

    tree.to_graphviz(
        filename=dot_name,
        shape="box"
    )

    try:
        subprocess.run(
            [
                "dot",
                "-Grankdir=LR",
                "-Tpng",
                dot_name,
                "-o",
                png_name
            ],
            check=True
        )

    except Exception as error:
        print(
            f"Could not make the PNG "
            f"(is Graphviz installed?): {error}"
        )