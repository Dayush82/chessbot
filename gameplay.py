import pygame
import chess
import random
import numpy as np

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 640

# Load images
chessboard_img = pygame.image.load("picture/chessboard.png")
chessboard_img = pygame.transform.scale(
    chessboard_img, (SCREEN_WIDTH, SCREEN_HEIGHT))

# Load and scale piece images
piece_images = {
    'K': pygame.transform.scale(pygame.image.load("picture/white_king.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'Q': pygame.transform.scale(pygame.image.load("picture/white_queen.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'R': pygame.transform.scale(pygame.image.load("picture/white_rook.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'B': pygame.transform.scale(pygame.image.load("picture/white_bishop.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'N': pygame.transform.scale(pygame.image.load("picture/white_knight.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'P': pygame.transform.scale(pygame.image.load("picture/white_pawn.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'k': pygame.transform.scale(pygame.image.load("picture/black_king.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'q': pygame.transform.scale(pygame.image.load("picture/black_queen.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'r': pygame.transform.scale(pygame.image.load("picture/black_rook.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'b': pygame.transform.scale(pygame.image.load("picture/black_bishop.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'n': pygame.transform.scale(pygame.image.load("picture/black_knight.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8))),
    'p': pygame.transform.scale(pygame.image.load("picture/black_pawn.png"), (int(SCREEN_WIDTH/8), int(SCREEN_HEIGHT/8)))
}





def chess_to_pixel(chess_coord):
    file, rank = chess.square_file(chess_coord), chess.square_rank(chess_coord)
    square_width = SCREEN_WIDTH / 8
    square_height = SCREEN_HEIGHT / 8
    x = file * square_width
    # Flip rank since screen coordinates start from the top-left
    y = (7 - rank) * square_height
    return (int(x), int(y))


def pixel_to_chess(pixel_coord):
    x, y = pixel_coord
    file = int(x / (SCREEN_WIDTH / 8))
    rank = 7 - int(y / (SCREEN_HEIGHT / 8))
    return chess.square(file, rank)


def draw_board_and_pieces(board, screen):
    screen.blit(chessboard_img, (0, 0))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_image = piece_images[piece.symbol()]
            screen.blit(piece_image, chess_to_pixel(square))
    pygame.display.flip()

def one_hot_encode_piece(piece_symbol):
    """Returns a one-hot encoded vector for the given chess piece."""
    # Define a mapping of piece symbols to their indices in the one-hot encoded vector
    piece_to_index = {
        'K': 0, 'Q': 1, 'R': 2, 'B': 3, 'N': 4, 'P': 5,
        'k': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'p': 11,
        '0': 12  # This represents an empty square
    }
    # Create a zero-filled vector of length 13
    encoding = [0] * 13
    # Set the appropriate position to 1
    encoding[piece_to_index[piece_symbol]] = 1
    return encoding

def board_to_matrix(board):
    """Converts a chess.Board object to a 3D matrix representation."""
    matrix = []
    for rank in reversed(range(8)):
        row = []
        for file in range(8):
            square = rank * 8 + file
            piece = board.piece_at(square)
            if piece:
                row.append(one_hot_encode_piece(piece.symbol()))
            else:
                row.append(one_hot_encode_piece('0'))  # For an empty square
        matrix.append(row)
    return matrix

def action_to_move(action, board):
    """
    Convert an action into a chess move.

    Parameters:
    - action: The action to convert.
    - board: The current chess board state.

    Returns:
    - move: A chess move or None if the action is illegal.
    """
    print(f"Action is a chess move: {action}")
    return action if action in board.legal_moves else None








def select_and_make_move(board, action):
    """
    Selects and makes a move on the board based on the chosen action.
    
    Parameters:
    - board: Current chess board state.
    - action: Chosen action from the DQN.
    
    Returns:
    - updated board after making the move.
    - game_over: a boolean indicating if the game is over.
    """
    if action:
        board.push(action)
    
    game_over = board.is_game_over()
    return board, game_over





def calculate_reward(action, board):
    """
    Calculate the reward based on the move made and the resulting board state.
    
    Parameters:
    - action: The action that was chosen.
    - board: The current chess board.
    
    Returns:
    - reward: Calculated reward.
    """
    # If action is None (invalid move was chosen), return a negative reward
    if action is None:
        return -10
    
    # Define piece values
    piece_values = {
    'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,  # Add a value for 'K'
    'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0   # Add a value for 'k'
    }

    
    # Initialize reward as 0
    reward = 0

    reward -= 0.1
    
    # Check for checkmate
    if board.is_checkmate():
        # If it's black's turn and black is in checkmate, then the agent made a bad move.
        reward = -1000 if board.turn == chess.BLACK else 1000
    
    # Check if a piece was captured
    elif board.piece_at(action.to_square):
        captured_piece_symbol = board.piece_at(action.to_square).symbol()
        reward = piece_values[captured_piece_symbol]
    
    # Check for checks
    elif board.is_check():
        reward = 50
    
    # Check for draw (you can adjust the reward value as needed)
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        reward = 20

    return reward

# The function is adjusted to use board state to detect captures. 
# Since the python-chess library isn't available in this environment, we won't be able to test it here.
# However, you can integrate this updated function into your code and test it in your local environment.

def is_game_over(board):
    return board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or board.can_claim_threefold_repetition()

def mask_illegal_moves(q_values, board):
    legal_moves = list(board.legal_moves)
    mask = np.full(4096, float('-inf'))  # Create a mask filled with very negative numbers
    for move in legal_moves:
        action = move.from_square * 64 + move.to_square
        mask[action] = q_values[0][action]
    return mask

def move_to_index(move):
    if isinstance(move, chess.Move):
        # Extracting from_square and to_square indices
        from_square = move.from_square
        to_square = move.to_square
        
        # Ensuring each square is represented with a unique index
        return from_square * 64 + to_square
    elif isinstance(move, (int, np.int64)):
        return move
    else:
        raise ValueError(f"Unsupported action type: {type(move)}")

    

def select_action(q_values, epsilon, legal_moves):
    """
    Select an action using epsilon-greedy policy.

    Parameters:
    - q_values: Q-values predicted by the DQN model.
    - epsilon: Exploration-exploitation trade-off parameter.
    - legal_moves: Legal moves available in the current state.

    Returns:
    - action: Selected action.
    """
    if random.random() < epsilon:
        # Exploration: randomly select a legal move
        return random.choice(list(legal_moves))


    else:
        # Exploitation: select the move with the highest Q-value
        try:
            legal_q_values = [q_values[0, move_to_index(move)] for move in legal_moves]
        except IndexError as e:
            print(f"Caught an IndexError: {str(e)}")
            print(f"Shape of q_values: {q_values.shape}")
            print(f"Some moves: {[move.uci() for move in legal_moves[:5]]}")  # Print first 5 moves
            print(f"Some indices: {[move_to_index(move) for move in legal_moves[:5]]}")  # Print first 5 indices
            raise  # This will still stop the program, but now with prints before it
        
        return legal_moves[np.argmax(legal_q_values)]

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(rewards, losses, save_path='training_plot.png'):
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    
    # Plotting rewards
    axs[0].plot(rewards, label='Reward per Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend(loc='upper left')
    
    # Compute and plot rolling average of rewards
    window_size = 100  # number of episodes for average
    rolling_avg = [np.mean(rewards[i-window_size:i]) for i in range(window_size, len(rewards))]
    axs[0].plot(range(window_size, len(rewards)), rolling_avg, label=f'Rolling Average (window={window_size})')
    
    # Plotting losses
    axs[1].plot(losses, label='Loss per Training Step', color='r')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper left')
    
    # Save plot to file
    plt.savefig(save_path)
    
    # Show plot
    #plt.show()


import chess
import time

def visualize_moves(moves):
    """
    Visualize a series of chess moves.

    Parameters:
    - moves: A list of UCI strings representing moves.
    """
    board = chess.Board()
    
    for move_uci in moves:
        # Convert the UCI string to a move object
        move = chess.Move.from_uci(move_uci)
        
        # Check if the move is legal
        if move in board.legal_moves:
            # Apply the move to update the board
            board.push(move)
            
            # Visualize the updated board
            print(board)
            
            # Optional: Introduce a delay for better visualization
            time.sleep(1)
        else:
            print(f"Move {move_uci} is illegal. Skipping.")
            continue







board = chess.Board()
print("Board state before move:")
print(board)
matrix = board_to_matrix(board)
for row in matrix:
    print(row)
