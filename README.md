# Pirate Intelligent Agent

## Overview

This project implements a deep Q-learning intelligent agent for a treasure hunt game. The agent controls a pirate non-player character that must navigate an 8×8 maze, avoid obstacles, and reach a treasure located in the bottom-right corner.

The pirate learns through reinforcement learning rather than following a hard-coded route. During training, the agent explores different movements, receives rewards or penalties from the environment, stores its experiences, and uses a neural network to estimate which action is most likely to lead to the treasure.

This project was completed for CS 370: Current and Emerging Trends in Computer Science at Southern New Hampshire University.

## Project Goal

The goal of the intelligent agent is to learn an effective path through the maze and consistently reach the treasure.

The pirate can perform four actions:

* Move left
* Move up
* Move right
* Move down

The environment provides feedback through the following reward structure:

* Reaching the treasure: positive reward
* Attempting to enter an occupied cell: penalty
* Attempting to move beyond the maze boundary: penalty
* Moving to an adjacent open cell: small penalty to discourage unnecessary wandering

By maximizing cumulative reward, the agent learns to avoid invalid movements and select efficient routes toward the treasure.

## Deep Q-Learning Approach

The project uses a Deep Q-Network, or DQN, to approximate the expected reward associated with each possible action.

At each step, the agent:

1. Observes the current maze state.
2. Determines the valid actions available from its current position.
3. Chooses between exploration and exploitation.
4. Performs the selected action.
5. Receives the next state, reward, and game status.
6. Stores the experience in replay memory.
7. Trains the neural network using a batch of previous experiences.
8. Repeats the process until it reaches the treasure or loses the episode.

An experience is stored in the following form:

```text
Current state, action, reward, next state, game status
```

These stored experiences allow the agent to learn from both recent and earlier decisions rather than relying only on the most recent movement.

## Exploration and Exploitation

The agent uses an epsilon-greedy strategy to balance exploration and exploitation.

During exploration, the pirate selects a random valid action. This helps it discover paths that it has not previously attempted.

During exploitation, the pirate uses the neural network to predict the Q-value of each action and selects the valid action with the highest estimated reward.

The exploration rate begins at a high value and gradually decreases during training:

```python
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
```

This allows the agent to explore heavily during early training and rely more on its learned policy as performance improves.

## Neural Network Architecture

The neural network is built with TensorFlow and Keras.

The model contains:

* An input layer based on the flattened 8×8 maze state
* Two fully connected hidden layers
* PReLU activation functions
* An output layer with four values, one for each possible movement
* The Adam optimizer
* Mean squared error as the loss function

```python
def build_model(maze):
    model = Sequential()
    model.add(Dense(maze.size, input_shape=(maze.size,)))
    model.add(PReLU())
    model.add(Dense(maze.size))
    model.add(PReLU())
    model.add(Dense(num_actions))
    model.compile(optimizer="adam", loss="mse")
    return model
```

Each output represents the predicted Q-value of moving left, up, right, or down from the current state.

## Experience Replay and Target Network

Experience replay stores previous state transitions and samples them in batches during training. Reusing past experiences helps reduce the correlation between consecutive movements and improves training stability.

The implementation also uses a target network. The target network is a copy of the primary model whose weights are updated periodically. It provides more stable target Q-values while the primary network is being trained.

```python
target_update_freq = 50
```

## Training Completion

The training process tracks recent wins and calculates a rolling win rate. Epsilon decreases as the agent gains experience, and the target network is updated at regular intervals.

Training stops early when the agent reaches the required win rate and passes the completion check from all valid starting positions.

The completed model is also tested from the designated starting position in the top-left corner of the maze.

## Technologies Used

* Python
* Jupyter Notebook
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Deep Q-learning
* Experience replay
* Epsilon-greedy exploration

## Project Files

```text
Hall_Shauntel_ProjectTwo.ipynb
TreasureMaze.py
GameExperience.py
requirements.txt
README.md
```

### `Hall_Shauntel_ProjectTwo.ipynb`

Contains the maze configuration, neural-network model, deep Q-learning training algorithm, completion check, and game test.

### `TreasureMaze.py`

Defines the maze environment, available actions, state updates, rewards, valid movements, and game status.

### `GameExperience.py`

Implements experience replay, stores state transitions, predicts Q-values, and prepares training batches.

### `requirements.txt`

Lists the Python packages required to run the project.

The provided Python environment files should not be modified.

## Running the Project

1. Clone or download the repository.

```bash
git clone <https://github.com/Shauntel92/CS370-Current-Emerging-Trends-in-CS.git>
cd <CS370-Current-Emerging-Trends-in-CS>
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Start Jupyter Notebook.

```bash
jupyter notebook
```

4. Open:

```text
Hall_Shauntel_ProjectTwo.ipynb
```

5. Select **Run All** to build the model, train the agent, run the completion check, and test the pirate from the top-left starting position.

Training may take several minutes depending on the available hardware.

## CPU Configuration

The notebook disables GPU execution because some hosted environments may not provide the required CUDA or BLAS support.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

This allows TensorFlow to run the neural-network calculations on the CPU.

## Key Learning Outcomes

This project demonstrates:

* How reinforcement learning can solve a pathfinding problem
* How rewards and penalties influence agent behavior
* How Q-values represent expected future rewards
* How neural networks approximate Q-values
* How exploration and exploitation work together
* How experience replay improves learning stability
* How target networks support deep Q-learning
* How an intelligent agent improves through repeated interaction with an environment

## Author

**Shauntel Hall**

Computer Science
Southern New Hampshire University


