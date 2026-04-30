---
title: Minesweeper RL Environment
emoji: 💣
colorFrom: purple
colorTo: orange
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
    - openenv
    - reinforcement-learning
    - minesweeper
    - curriculum-learning
---

# Minesweeper RL Environment

A reinforcement learning environment for the classic Minesweeper game, built on the OpenEnv framework. This environment enables training and evaluating RL agents on Minesweeper with built-in curriculum learning and support for various agent architectures (MLP, CNN, LLM-based).

## Quick Start

The simplest way to use the Minesweeper environment is through the `MinesweeperEnv` class:

```python
from minesweeper_env import MinesweeperAction, MinesweeperEnv

try:
    # Create environment from Docker image
    env = MinesweeperEnv.from_docker_image("minesweeper-env:latest")

    # Reset with default parameters (8x8 grid, 10 mines)
    result = env.reset()
    print(f"Game started: {result.observation.message}")

    # Simple random agent example
    import numpy as np

    while not result.done:
        # Get available actions (unrevealed cells)
        action_mask = result.observation.action_mask
        available_actions = np.where(action_mask)[0]

        if len(available_actions) == 0:
            break

        # Choose a random valid action
        action_idx = np.random.choice(available_actions)
        row, col = divmod(int(action_idx), result.observation.n)

        # Step the environment
        result = env.step(MinesweeperAction(row=row, col=col))
        print(f"Action: ({row},{col}) → {result.observation.message}")
        print(f"Reward: {result.reward:.2f}")

finally:
    # Always clean up
    env.close()
```

That's it! The `MinesweeperEnv.from_docker_image()` method handles:

- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t minesweeper-env:latest -f server/Dockerfile .
```

## Environment Features

### Curriculum Learning

The environment includes built-in curriculum learning through the `solve_tiles` parameter:

- **solve_tiles**: Number of safe tiles automatically revealed at the start of each episode
- **Curriculum progression**: When the agent wins 3 consecutive games, `solve_tiles` decreases by 1
- **Full difficulty**: At `solve_tiles=0`, the agent plays with no hints

```python
# Start with easier difficulty (5 tiles pre-revealed)
result = env.reset(solve_tiles=5)

# Force specific difficulty level
result = env.reset(n=8, mines=10, solve_tiles=3)
```

### Observation Space

**MinesweeperObservation** provides multiple views of the game state:

- `board`: 2D list of integers (0-9)
    - 0-8: revealed safe cell (adjacency count)
    - 9: unrevealed cell (both safe and mines)
- `n`: grid side length
- `mines_count`: total mines on board
- `unrevealed_count`: number of unrevealed cells
- `status`: game status (0=ONGOING, 1=WON, 2=LOST)
- `message`: human-readable step summary

**Agent-ready views**:

```python
obs = result.observation

# MLP/DQN input: (n*n,) float32 in [0.0, 1.0]
mlp_input = obs.flat

# CNN input: (1, n, n) float32 in [0.0, 1.0]
cnn_input = obs.spatial

# Action masking: (n*n,) bool - True where click is valid
action_mask = obs.action_mask
```

### Action Space

**MinesweeperAction**: Simple (row, col) coordinates

```python
from minesweeper_env import MinesweeperAction

# Click on cell (2, 3)
action = MinesweeperAction(row=2, col=3)
result = env.step(action)
```

### Reward Structure

Rewards are designed to encourage efficient play:

- **0.0**: Re-clicking revealed cells or hitting mines
- **new_cells/total_safe**: Safe reveal (proportional to progress)
- **1.0**: Winning the game (terminal bonus)

**Example rewards**:

- Reveal 3 new cells on 56-cell board → 3/56 ≈ 0.05
- Win game → 1.0
- Hit mine → 0.0

### Game Difficulty Levels

Standard Minesweeper difficulties for reference:

| Level        | Grid Size | Mines | Mine Density |
| ------------ | --------- | ----- | ------------ |
| Beginner     | 9×9       | 10    | 0.123        |
| Intermediate | 16×16     | 40    | 0.156        |
| Expert       | 16×30     | 99    | 0.206        |

Custom configurations:

```python
# Easy 6x6 with 5 mines (density: 0.138)
result = env.reset(n=6, mines=5)

# Hard 10x10 with 20 mines (density: 0.200)
result = env.reset(n=10, mines=20)
```

## Advanced Usage

### Connecting to an Existing Server

If you already have a Minesweeper server running:

```python
from minesweeper_env import MinesweeperEnv

# Connect to existing server
env = MinesweeperEnv(base_url="http://localhost:8000")
result = env.reset(n=8, mines=10)
result = env.step(MinesweeperAction(row=0, col=0))
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Using the Context Manager

```python
from minesweeper_env import MinesweeperAction, MinesweeperEnv

# Automatic connection management
with MinesweeperEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    while not result.done:
        # Simple policy: click first available cell
        action_mask = result.observation.action_mask
        action_idx = action_mask.argmax()
        row, col = divmod(int(action_idx), result.observation.n)
        result = env.step(MinesweeperAction(row=row, col=col))
```

The client uses WebSocket connections for:

- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections:

```python
from minesweeper_env import MinesweeperAction, MinesweeperEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with MinesweeperEnv(base_url="http://localhost:8000") as env:
        result = env.reset(n=6, mines=6)
        while not result.done:
            action_mask = result.observation.action_mask
            if action_mask.any():
                action_idx = action_mask.argmax()
                row, col = divmod(int(action_idx), result.observation.n)
                result = env.step(MinesweeperAction(row=row, col=col))
        return client_id, result.reward

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

### Async Usage

For async training loops:

```python
import asyncio
from minesweeper_env import MinesweeperAction, MinesweeperEnv

async def play_game():
    async with MinesweeperEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(n=8, mines=10, solve_tiles=2)

        while not result.done:
            # Simple policy logic here
            action_mask = result.observation.action_mask
            action_idx = action_mask.argmax()
            row, col = divmod(int(action_idx), result.observation.n)

            result = await env.step(MinesweeperAction(row=row, col=col))
            print(f"Step {result.observation.step_count}: Reward {result.reward:.2f}")

        state = await env.state  # Get curriculum state
        return result.reward

# Run async game
reward = asyncio.run(play_game())
```

### Accessing Curriculum State

```python
result = env.reset(solve_tiles=5)
state = env.state  # MinesweeperState object

print(f"Episode: {state.episode_id}")
print(f"Grid size: {state.n}x{state.n}")
print(f"Mine density: {state.mine_density}")
print(f"Safe cells: {state.safe_cells}")
print(f"Current solve_tiles: {state.solve_tiles}")
```

## Deployment Options

### Hugging Face Spaces

Deploy your OpenEnv environment to Hugging Face Spaces:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The deployed space includes:

- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

### Local Development

Run the server locally:

```bash
# With auto-reload for development
uvicorn server.app:app --reload

# Production with multiple workers
uvicorn server.app:app --workers 4

# Or use the module entry point
python -m minesweeper_env.server.app --port 8000
```

## Training RL Agents

### MLP/DQN Agent Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from minesweeper_env import MinesweeperAction, MinesweeperEnv

class DQN(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        input_size = n * n
        hidden_size = 128
        output_size = n * n  # One Q-value per cell

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Training loop
with MinesweeperEnv(base_url="http://localhost:8000") as env:
    model = DQN(n=8)
    optimizer = optim.Adam(model.parameters())

    result = env.reset()
    while not result.done:
        obs = result.observation

        # Convert observation to tensor
        state_tensor = torch.from_numpy(obs.flat).float()

        # Get Q-values and mask invalid actions
        with torch.no_grad():
            q_values = model(state_tensor)
            q_values[~obs.action_mask] = -float('inf')
            action_idx = q_values.argmax()

        row, col = divmod(int(action_idx), obs.n)
        result = env.step(MinesweeperAction(row=row, col=col))

        # Store transition and train (simplified)
        # ... training logic here ...
```

### CNN Agent Example

```python
import torch
import torch.nn as nn

class CNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 8 * 8)  # For 8x8 grid

    def forward(self, x):
        # x: (batch, 1, 8, 8)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Usage
obs = result.observation
spatial_input = torch.from_numpy(obs.spatial).float().unsqueeze(0)  # Add batch dim
q_values = model(spatial_input)
```

## Environment Details

### Complete API Reference

**MinesweeperAction**: Single cell click

- `row` (int): Row coordinate (0 to n-1)
- `col` (int): Column coordinate (0 to n-1)

**MinesweeperObservation**: Everything the agent sees

- `board` (List[List[int]]): 2D game board
- `n` (int): Grid side length
- `mines_count` (int): Total mines on board
- `unrevealed_count` (int): Number of unrevealed cells
- `status` (int): Game status (0=ONGOING, 1=WON, 2=LOST)
- `message` (str): Human-readable step summary
- `done` (bool): Episode termination flag
- `reward` (float): Reward for current step

**MinesweeperState**: Server-side episode metadata

- `episode_id` (str): Unique episode identifier
- `step_count` (int): Steps taken in current episode
- `n` (int): Grid side length
- `mines_count` (int): Number of mines
- `solve_tiles` (int): Current curriculum difficulty level
- `mine_density` (float): Computed mine density
- `safe_cells` (int): Total safe cells

### Game Status Codes

```python
from minesweeper_env import GameStatus

GameStatus.ONGOING  # 0 - Game in progress
GameStatus.WON      # 1 - Game won
GameStatus.LOST     # 2 - Game lost
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/minesweeper_environment.py
```

This verifies:

- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Unit Testing

The environment includes comprehensive type hints and follows PEP 8 standards. Run tests with:

```bash
# Install dev dependencies
uv sync --dev

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest --cov=minesweeper_env --cov-report=term-missing
```

## Project Structure

```
minesweeper_env/
├── .dockerignore              # Docker build exclusions
├── __init__.py               # Module exports
├── README.md                 # This file
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # Project metadata and dependencies
├── uv.lock                   # Locked dependencies (generated)
├── client.py                 # MinesweeperEnv client
├── models.py                 # Action and Observation models
└── server/
    ├── __init__.py           # Server module exports
    ├── game_engine.py        # Core Minesweeper game logic
    ├── environment.py        # RL environment wrapper
    ├── app.py                # FastAPI application
    └── Dockerfile            # Container image definition
```

## Support & Resources

- **OpenEnv Framework**: Base framework for RL environments
- **Pydantic**: Data validation and serialization
- **FastAPI**: Web server and WebSocket support
- **NumPy**: Array operations and observation processing

For issues or questions, refer to the OpenEnv documentation or create an issue in the repository.

## License

This project is licensed under the BSD-style license. See the LICENSE file for details.
