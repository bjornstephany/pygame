from typing import List, Optional, Tuple

from halma import *
from treelib import Tree

SCORE_TOTAL = 100.0

tree: Optional[Tree] = None
node_counter = 0

# Helper functions
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

# Evaluation Function (Heuristic approach)
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

# Max^n search function with shallow pruning
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

        if parent_id is not None and ply < 2:

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
        5,
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

    png_name = f"Team3_Tree.png"

    save_tree_png(png_name)


def save_tree_png(png_name: str) -> None:
    import pygame

    if tree is None:
        return

    pygame.font.init()

    font = pygame.font.Font(None, 22)

    nodes = list(tree.all_nodes_itr())

    node_padding_x = 14
    node_height = 34
    x_gap = 80
    y_gap = 18
    margin = 30

    node_width = max(font.size(str(node.tag))[0] for node in nodes)
    node_width += node_padding_x * 2

    positions = {}
    next_leaf_y = 0

    def place_node(node_id, depth):
        nonlocal next_leaf_y

        children = tree.children(node_id)

        if len(children) == 0:
            y = next_leaf_y
            next_leaf_y += node_height + y_gap
        else:
            child_ys = [
                place_node(child.identifier, depth + 1)
                for child in children
            ]
            y = (child_ys[0] + child_ys[-1]) // 2

        x = depth * (node_width + x_gap)
        positions[node_id] = (x, y)

        return y

    place_node(tree.root, 0)

    max_x = max(x for x, unused_y in positions.values()) + node_width
    max_y = max(y for unused_x, y in positions.values()) + node_height

    surface = pygame.Surface(
        (max_x + margin * 2, max_y + margin * 2)
    )
    surface.fill((255, 255, 255))

    line_color = (80, 80, 80)
    box_color = (245, 247, 251)
    border_color = (30, 39, 55)
    text_color = (30, 39, 55)

    for node in nodes:
        parent_id = node.predecessor(tree.identifier)

        if parent_id is None:
            continue

        parent_x, parent_y = positions[parent_id]
        child_x, child_y = positions[node.identifier]

        start = (
            margin + parent_x + node_width,
            margin + parent_y + node_height // 2
        )
        end = (
            margin + child_x,
            margin + child_y + node_height // 2
        )

        pygame.draw.line(surface, line_color, start, end, 2)

    for node in nodes:
        x, y = positions[node.identifier]
        rect = pygame.Rect(
            margin + x,
            margin + y,
            node_width,
            node_height
        )

        pygame.draw.rect(surface, box_color, rect)
        pygame.draw.rect(surface, border_color, rect, 2)

        text = font.render(str(node.tag), True, text_color)
        surface.blit(text, text.get_rect(center=rect.center))

    pygame.image.save(surface, png_name)
