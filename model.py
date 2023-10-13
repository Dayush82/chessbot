import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, load_model
import os

def create_dqn_model():
    """
    Create and return the DQN model for the chess game.
    """
    
    # Input: board state
    input_board = layers.Input(shape=(8, 8, 13), name="BoardInput")
    
    # Convolutional layers
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="Conv1")(input_board)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="Conv2")(x)
    
    # Flatten and dense layers
    x = layers.Flatten(name="Flatten")(x)
    x = layers.Dense(256, activation='relu', name="Dense1")(x)
    
    # Output layer: Q-values for each possible move
    output_moves = layers.Dense(64*64, activation='linear', name="OutputMoves")(x)
    
    # Create model
    model = models.Model(inputs=input_board, outputs=output_moves, name="ChessDQN")
    
    return model


def load_dqn_model(path):
    if os.path.exists(path):
        return load_model(path)
    else:
        return create_dqn_model()



from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def save_dqn_model(model, path):
    model.save(path)
dqn_model = create_dqn_model()

dqn_model.save('models/my_model.keras')


optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)


dqn_model.compile(optimizer=optimizer,
              loss='mse')


def prepare_model():
    model = create_dqn_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Ensure the returned, compiled model is used for training
dqn_model = prepare_model()


print("Model compiled!")


#dqn_model.save_weights('models/dqn_model.h5')

# Create the model for visualization
#dqn_model = create_dqn_model()
#dqn_model.summary()
