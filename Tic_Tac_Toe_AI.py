import random

# Tic-Tac-Toe board
board = [" " for _ in range(9)]

# Print the Tic-Tac-Toe board
def print_board(board):
    for i in range(0, 9, 3):
        print(board[i], "|", board[i + 1], "|", board[i + 2])
        if i < 6:
            print("---------")

# Check for a win
def check_win(board, player):
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]             # Diagonals
    ]
    for combo in win_combinations:
        if all(board[i] == player for i in combo):
            return True
    return False

# Check for a draw
def check_draw(board):
    return " " not in board

# Generate a list of available moves
def get_available_moves(board):
    return [i for i, cell in enumerate(board) if cell == " "]

# Minimax algorithm
def minimax(board, depth, is_maximizing):
    if check_win(board, "X"):
        return -1
    elif check_win(board, "O"):
        return 1
    elif check_draw(board):
        return 0

    if is_maximizing:
        max_eval = -float("inf")
        for move in get_available_moves(board):
            board[move] = "O"
            eval = minimax(board, depth + 1, False)
            board[move] = " "
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        for move in get_available_moves(board):
            board[move] = "X"
            eval = minimax(board, depth + 1, True)
            board[move] = " "
            min_eval = min(min_eval, eval)
        return min_eval

# AI's move using minimax
def ai_move(board):
    best_move = None
    best_eval = -float("inf")
    for move in get_available_moves(board):
        board[move] = "O"
        eval = minimax(board, 0, False)
        board[move] = " "
        if eval > best_eval:
            best_eval = eval
            best_move = move
    return best_move

# Main game loop
print("Welcome to Tic-Tac-Toe!")
while True:
    print_board(board)
    move = int(input("Enter your move (0-8): "))
    if board[move] == " ":
        board[move] = "X"
        if check_win(board, "X"):
            print_board(board)
            print("You win!")
            break
        elif check_draw(board):
            print_board(board)
            print("It's a draw!")
            break
        ai_move_idx = ai_move(board)
        board[ai_move_idx] = "O"
        if check_win(board, "O"):
            print_board(board)
            print("AI wins!")
            break
    else:
        print("Invalid move. Try again.")
