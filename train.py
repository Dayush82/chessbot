import chess
import chess.svg
import pygame
import numpy as np

import random
from tensorflow import keras
from tensorflow.keras.models import load_model
import os 

from model import create_dqn_model, ReplayBuffer, dqn_model, prepare_model
from gameplay import board_to_matrix, select_and_make_move, calculate_reward, action_to_move, mask_illegal_moves, draw_board_and_pieces, move_to_index, select_action, plot_training_progress,visualize_moves


# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8
GAMMA = 0.99
NUM_EPISODES = 1000  # Number of games to play
MAX_MOVES_PER_GAME = 100  # Maximum number of moves for each game
epsilon = 1.0
epsilon_decay = 0.995  # Example value, can be adjusted
min_epsilon = 0.01  # Minimum value for epsilon to prevent it from going to 0
episode = 0
SAVE_FREQUENCY = 32
total_reward = 0

plot_rewards = []  # to store rewards for each episode
plot_losses = []


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Initialize screen and clock
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess")
clock = pygame.time.Clock()

# Replay buffer
replay_buffer = ReplayBuffer(10000)
BATCH_SIZE = 32

# Training loop
game_number = 0
move_number = 0


dqn_model = prepare_model()

#outer loop
for _ in range(NUM_EPISODES):
    reward = 0
    board = chess.Board()
    #inner loop 
    for move in range(MAX_MOVES_PER_GAME):
        print("\nEpisode:", episode+1, "Move number:", move+1)
        
        current_state = board_to_matrix(board)
        current_state = np.expand_dims(current_state, axis=0)
        
        legal_moves = list(board.legal_moves)
        illegal_moves = set()
        if os.path.exists('my_model.keras'):
            dqn_model = load_model('my_model.keras')
        else:
            dqn_model = create_dqn_model()
       
        q_values = dqn_model.predict(current_state)
        action = select_action(q_values, epsilon,legal_moves) 
        move = action_to_move(action, board)

        if move not in board.legal_moves:
            # Penalize illegal move
            reward = -100
            
            # Choose a random legal move as a fallback
            move = random.choice(list(board.legal_moves))
            action = move_to_index(move)

        else:
            # Reward can be calculated based on the move's outcome
            reward = calculate_reward(action, board)


        next_board, game_over = select_and_make_move(board, action)

       

        reward = calculate_reward(action, board)


        total_reward +=reward



        print(f"Move: {move.uci()}", move, "Reward:", total_reward)
        print("Board state:")
        print(next_board)

        # Check if the indices are within the bounds
        #for move in legal_moves:
            #idx = move_to_index(move)
            #if idx >= np.shape(q_values)[0] or idx < 0:  # Assuming q_values is a 1D array
                #print(f"Index {idx} out of bounds for move {move.uci()}")



        replay_buffer.push(current_state, action, reward, board_to_matrix(next_board), game_over)

        # Update the board state
        board = next_board
        
        # Check for game termination
        if board.is_game_over() or move == MAX_MOVES_PER_GAME - 1:
            break

        
        # Training
        if len(replay_buffer) >= BATCH_SIZE:
            # Sample a batch from the replay buffer
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states = np.array(states).squeeze()
            next_states = np.array(next_states).squeeze()
            q_values = dqn_model.predict(states)
            next_q_values = dqn_model.predict(next_states)
            
            # Compute Q targets
            max_next_q_values = np.max(next_q_values, axis=1)
            GAMMA = np.array(GAMMA)
            print("reward :", reward)
            print("max_next_q_values", max_next_q_values)
            print("dones", dones)
            print("current buffer", len(replay_buffer))
            dones_array = np.array(dones, dtype=np.float32)  # Convert dones tuple to a numpy array
            q_targets = rewards + GAMMA * max_next_q_values * (1 - dones_array)
            print("\nTraining the model...")
            print("Actions before error:", actions)

            print(f"Actions: {actions}")
            print(f"Actions types: {[type(a) for a in actions]}")

            # Convert actions to indices
            try:
                action_indices = [move_to_index(a) for a in actions]
            except Exception as e:
                print(f"Error converting actions to indices: {e}")
                print(f"Actions causing error: {actions}")
                raise e  # re-raise the exception to halt the program and inspect the error

            # Check and print the converted action indices
            print(f"Action indices: {action_indices}")


            action_indices = [move_to_index(move) for move in actions]

            action_indices = np.array(action_indices)



            batch_indices = np.arange(BATCH_SIZE)

            print(f"Actions shape: {np.shape(actions)}, type: {type(actions)}")
            print(f"q_targets shape: {np.shape(q_targets)}, type: {type(q_targets)}")
            assert np.all(actions is not None), "None value detected in actions!"
            assert np.all(q_targets is not None), "None value detected in q_targets!"
            assert q_values.shape[0] == BATCH_SIZE, f"Mismatched batch sizes: q_values has {q_values.shape[0]} rows, expected {BATCH_SIZE}"

            if isinstance(actions, tuple):  # or alternatively: if not isinstance(actions, np.ndarray):
                actions = np.array(action_indices, dtype=int)

            # Additional checks to ensure actions are valid indices
            assert actions.shape == (BATCH_SIZE,), "Actions shape mismatch!"
            assert np.all(action_indices >= 0) and np.all(action_indices < q_values.shape[1]), "Invalid action indices detected!"



            q_values[range(BATCH_SIZE), actions] = q_targets
            

            dqn_model = prepare_model()
            
            # Train the DQN model
            loss = dqn_model.train_on_batch(states, q_values)

            visualize_moves(move)
            print('Loss:', loss)
            if not os.path.exists('plots'):
                os.makedirs('plots')
            # After each episode or periodically during training
            plot_rewards.append(reward)
            plot_losses.append(loss)
            plot_training_progress(rewards, loss, save_path=f'plots/episode_{game_number}.png')





        epsilon = max(epsilon * epsilon_decay, min_epsilon)

        # For visualization, display the board and introduce a delay
        draw_board_and_pieces(board, screen)
        pygame.time.wait(1000)  # 500 milliseconds delay
        print(f"Game number {game_number + 1} finished")
        if episode % SAVE_FREQUENCY == 0:
            save_path = 'models/my_model.keras'
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dqn_model.save(save_path)


    # Increment episode count after all moves for the current game are made
    game_number += 1

