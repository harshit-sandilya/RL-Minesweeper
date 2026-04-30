# Deep Q-Networks (DQN) — From Idea to Implementation

This document explains the Deep Q-Network (DQN) algorithm in a way that you can:

- Understand the intuition behind it.
- See all the important equations clearly.
- Implement it from scratch in PyTorch (or any DL framework) inside an RL codebase.

It assumes you know basic Python, a bit of PyTorch, and high‑school maths.

The description is **general** (not tied to Minesweeper), but at the end there is a short section on how to plug a DQN agent into a Minesweeper environment and into a typical `rl/` code layout.

---

## 1. Problem Setting: What DQN Tries to Solve

DQN solves **value-based reinforcement learning problems** where:

- An agent interacts with an environment over time.
- At each time step \(t\), the agent sees a **state** \(s_t\), chooses an **action** \(a_t\), receives a **reward** \(r_t\), and moves to the next state \(s_{t+1}\).[web:7]
- The goal is to **maximize the long‑term total reward** (called the *return*).[web:7]

This setting is modelled as a **Markov Decision Process (MDP)**:

- State space: \(\mathcal{S}\)
- Action space: \(\mathcal{A}\)
- Transition dynamics: \(P(s_{t+1} \mid s_t, a_t)\)
- Reward function: \(R(s_t, a_t)\)
- Discount factor: \(\gamma \in [0,1)\), which controls how much we care about future rewards.

The **return** starting at time \(t\) is

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots
\]

The agent’s behaviour is a **policy** \(\pi(a \mid s)\): given a state, how likely it is to choose each action.

---

## 2. Q-Learning Recap (Without Deep Learning)

Before DQN, there was **tabular Q-learning**.

### 2.1. Action-Value Function

The **action-value function** (Q-function) of a policy \(\pi\) is

\[
Q^{\pi}(s,a) = \mathbb{E}[G_t \mid s_t = s, a_t = a, \pi]  \quad (1)
\]

This is the expected return if you start from state \(s\), take action \(a\), and then follow policy \(\pi\).[web:7]

The **optimal** Q-function is

\[
Q^{*}(s,a) = \max_{\pi} Q^{\pi}(s,a).  \quad (2)
\]

If you know \(Q^{*}\), the optimal policy is simply

\[
\pi^{*}(s) = \arg\max_{a} Q^{*}(s,a).  \quad (3)
\]

### 2.2. Bellman Optimality Equation

\(Q^{*}\) satisfies the **Bellman optimality equation**:

\[
Q^{*}(s,a) = \mathbb{E}_{s' \sim P}\left[ r(s,a) + \gamma \max_{a'} Q^{*}(s', a') \right].  \quad (4)
\]

In tabular Q-learning, we store one value per \((s,a)\) pair and update it using sampled transitions.

Given a transition \((s_t, a_t, r_t, s_{t+1})\), the **tabular Q-learning update** is

\[
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \Big( r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \Big),  \quad (5)
\]

where \(\alpha\) is the learning rate.[web:14]

This works when the number of states and actions is small (you can store a table). But for large or continuous state spaces (e.g., images, big boards), tabular methods break.

---

## 3. From Q-Learning to Deep Q-Networks

The key idea of DQN is:

> **Replace the Q-table with a deep neural network that approximates \(Q(s,a)\).**[web:7][web:8]

So instead of storing \(Q(s,a)\) in a table, we have a neural network with parameters \(\theta\):

\[
Q(s,a; \theta) \approx Q^{*}(s,a).  \quad (6)
\]

### 3.1. Why a Neural Network?

- A network can handle **large, high‑dimensional states** (images, big grids, etc.).[web:7]
- It can **generalize**: if it sees similar states, it can produce similar Q-values.
- We can train it using **gradient descent** on a suitable loss function.[web:14]

However, directly plugging a neural network into Q-learning is unstable:

- Samples are **correlated** in time.
- Targets are **moving** because they depend on the same network we are updating.

DQN introduces two key techniques to stabilize learning:[web:8][web:14]

1. **Experience Replay** (replay buffer).
2. **Target Network** (a slowly updated copy of the main network).

We will explain both in detail.

---

## 4. DQN Components

A basic DQN implementation consists of these pieces:

1. **Q-network**: a neural network \(Q(s,a; \theta)\).
2. **Target Q-network**: \(Q(s,a; \theta^{-})\), a copy of the Q-network, updated more slowly.[web:5][web:14]
3. **Replay buffer**: stores past transitions \((s,a,r,s',\text{done})\).[web:11][web:14]
4. **Action selection policy**: typically \(\epsilon\)-greedy.
5. **Optimization loop**: sample minibatches, compute TD targets, compute loss, backpropagate.

We now go through each part.

---

## 5. Q-Network Architecture

The Q-network takes a state \(s\) as input and outputs one Q-value per action.

If the action space is **discrete** (e.g., 4 actions), the network is a function

\[
\text{NN}: s \mapsto (Q(s,a_1; \theta), Q(s,a_2; \theta), \dots, Q(s,a_{|\mathcal{A}|}; \theta)).  \quad (7)
\]

In code, this usually means:

- Input: \(s\) (tensor of shape like `(batch_size, state_dim)` or an image tensor).
- Output: a tensor of shape `(batch_size, num_actions)`.

Example MLP for low‑dim state (e.g., Minesweeper features):

- Linear layer: `state_dim → 128`, ReLU.
- Linear layer: `128 → 128`, ReLU.
- Output layer: `128 → num_actions` (no activation).

You can adapt this to CNNs for image inputs as done in the original Atari DQN paper.[web:7]

---

## 6. Experience Replay (Replay Buffer)

### 6.1. Problem: Correlated Data and Non‑Stationarity

If we train the network on consecutive transitions \((s_t,a_t,r_t,s_{t+1})\), the samples are highly correlated.[web:11]

Neural networks train best when data points are **independent and identically distributed (i.i.d.)**.

Also, because we are constantly updating the network, the target values we are trying to match change over time, which makes learning unstable.[web:11]

### 6.2. Solution: Replay Buffer

DQN uses a **replay buffer** (also called experience replay) to break these correlations.[web:8][web:11]

- Every time the agent steps in the environment, it stores a transition:

  \[
  (s_t, a_t, r_t, s_{t+1}, \text{done}_t)
  \]

- These are appended to a large buffer (e.g., capacity of 100k or 1M transitions).
- For learning, we sample **random minibatches** from this buffer instead of always taking the most recent transition.

Benefits:[web:11][web:14]

- Breaks temporal correlations.
- Reuses past experience many times → better data efficiency.
- Stabilizes training.

In code, this is typically a class with:

- `push(s, a, r, s_next, done)`
- `sample(batch_size) → (batch_s, batch_a, batch_r, batch_s_next, batch_done)`

---

## 7. Target Network

### 7.1. Problem: Moving Targets

In Q-learning, the target for a transition \((s,a,r,s')\) is

\[
y = r + \gamma \max_{a'} Q(s', a'; \theta).  \quad (8)
\]

If we use the *same* network (same parameters \(\theta\)) for both the prediction and the target, the target \(y\) changes every time we update \(\theta\).[web:3][web:6]

This makes the learning target drift and can lead to divergence.

### 7.2. Solution: Target Network

DQN maintains a **separate copy** of the network with parameters \(\theta^{-}\) called the **target network**.[web:5][web:8][web:14]

- The main network (online network) has parameters \(\theta\).
- The target network has parameters \(\theta^{-}\).
- During training, we compute the target using the target network:

\[
y = r + \gamma \max_{a'} Q(s', a'; \theta^{-}).  \quad (9)
\]

- Every **C steps** (a hyperparameter), we copy \(\theta\) into \(\theta^{-}\):

\[
\theta^{-} \leftarrow \theta.  \quad (10)
\]

This makes the target **piecewise constant**, so learning is more stable.[web:8][web:14]

(Some variants use a soft/Polyak update, but basic DQN uses hard copies every fixed number of steps.)[web:5]

---

## 8. DQN Loss Function

### 8.1. Temporal-Difference Target

For each sampled transition \((s,a,r,s',\text{done})\), we define the **TD target** \(y\) as

\[
y = \begin{cases}
  r & \text{if } \text{done} = 1 \\
  r + \gamma \max_{a'} Q(s', a'; \theta^{-}) & \text{otherwise.}
\end{cases}  \quad (11)
\]

This says:

- If the next state is terminal, the best we can do is just the immediate reward.
- Otherwise, we add the discounted value of the best action in the next state, as estimated by the **target network**.[web:3][web:14]

### 8.2. Squared TD Error Loss

Our network predicts \(Q(s,a; \theta)\). The **TD error** is

\[
\delta = y - Q(s,a; \theta).  \quad (12)
\]

DQN minimizes the **mean squared TD error** over a minibatch:

\[
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ (y - Q(s,a; \theta))^2 \right],  \quad (13)
\]

where \(\mathcal{D}\) is the replay buffer.[web:2][web:3]

In practice, we approximate this expectation by the average over a sampled batch of size `batch_size`.

Gradient descent (or Adam) updates the parameters \(\theta\) to reduce this loss.[web:14]

---

## 9. Exploration: Epsilon-Greedy Policy

DQN typically uses an **\(\epsilon\)-greedy** policy during training:

- With probability \(\epsilon\): choose a **random** action (explore).
- With probability \(1-\epsilon\): choose the **greedy** action

  \[
  a_t = \arg\max_{a} Q(s_t, a; \theta).  \quad (14)
  \]

\(\epsilon\) is often **annealed** over time:

- Start high (e.g., 1.0) to explore a lot.
- Linearly or exponentially decay to a smaller value (e.g., 0.1 or 0.01).

This balances exploration and exploitation while training.[web:14]

---

## 10. Full DQN Algorithm (Step-by-Step)

Below is the core DQN algorithm in words.

### 10.1. Initialization

1. Initialize **replay buffer** \(\mathcal{D}\) (empty).
2. Initialize **Q-network** with random parameters \(\theta\).
3. Initialize **target network** with parameters \(\theta^{-} = \theta\).
4. Set exploration parameter \(\epsilon\) (e.g., 1.0).

### 10.2. Main Training Loop

For each episode:

1. Reset environment, get initial state \(s_0\).
2. For each step in the episode:
   1. With probability \(\epsilon\) select a random action \(a_t\), otherwise select

      \[
      a_t = \arg\max_a Q(s_t, a; \theta).
      \]

   2. Execute \(a_t\) in the environment.
   3. Observe reward \(r_t\), next state \(s_{t+1}\), and done flag \(\text{done}_t\).
   4. Store transition \((s_t, a_t, r_t, s_{t+1}, \text{done}_t)\) in replay buffer \(\mathcal{D}\).
   5. If replay buffer has at least `batch_size` transitions:
      1. Sample a minibatch of transitions from \(\mathcal{D}\).
      2. For each transition in the batch, compute target \(y\) using equation (11).
      3. Compute predicted Q-values \(Q(s,a; \theta)\) for actions actually taken.
      4. Compute loss \(L(\theta)\) from equation (13).
      5. Take one gradient step on \(L(\theta)\) (e.g., using Adam optimizer).
   6. Periodically (every `target_update_freq` steps) update the target network: \(\theta^{-} \leftarrow \theta\).
   7. Decay \(\epsilon\) according to the schedule.
   8. If \(\text{done}_t\) is true, break the episode.

Repeat this process for many episodes.

During evaluation (test time), we usually set \(\epsilon\) to a very small value (e.g., 0.01) or 0, so the agent behaves greedily.

---

## 11. Typical Hyperparameters

Common DQN hyperparameters used in practice (inspired by Atari but adaptable):[web:7][web:8][web:14]

- Replay buffer capacity: `50_000` to `1_000_000` transitions.
- Min replay size before learning: e.g., `1_000` or `10_000`.
- Batch size: `32` or `64`.
- Discount factor \(\gamma\): around `0.99`.
- Learning rate: `1e-4` to `1e-3` (Adam).
- Target network update frequency: e.g., every `1_000` or `10_000` environment steps.
- \(\epsilon\)-greedy:
  - `epsilon_start = 1.0`
  - `epsilon_end = 0.1` or `0.01`
  - `epsilon_decay_steps` from `100_000` to `1_000_000` steps.

These should be tuned per environment, but they give a good starting point.

---

## 12. Implementation Guide (Code Structure)

Assume you have a project layout like:

```text
rl/
  agents/
  algorithms/
  common/
  networks/
```

A natural way to organize DQN is:

- `networks/dqn_network.py`: neural network definition.
- `common/replay_buffer.py`: replay buffer implementation.
- `algorithms/dqn.py`: the DQN update logic (learning algorithm, independent of a specific env).
- `agents/dqn_agent.py`: glue code that interacts with the environment (selects actions, stores transitions, calls `dqn.update()`), and logging.

Below is a conceptual breakdown.

### 12.1. Q-Network Module (`networks/dqn_network.py`)

Responsibilities:

- Define `class DQNNetwork(nn.Module)`.
- `forward(self, state)` returns Q-values for all actions.

Key points:

- Input shape should match your environment’s state representation (e.g., flattened board, CNN for images).
- Output dimension is `num_actions`.

### 12.2. Replay Buffer (`common/replay_buffer.py`)

Responsibilities:

- Store transitions `(s, a, r, s_next, done)`.
- Return random minibatches for training.

You can implement it using fixed‑size arrays or a `deque`.

### 12.3. DQN Algorithm Class (`algorithms/dqn.py`)

This class should:

- Hold:
  - Online Q-network (\(\theta\)).
  - Target Q-network (\(\theta^{-}\)).
  - Optimizer.
  - Hyperparameters (`gamma`, `batch_size`, `target_update_freq`, etc.).

- Provide a method like `update(replay_buffer, logger, global_step)` that:
  1. Samples a batch.
  2. Computes targets with the target network.
  3. Computes the loss.
  4. Performs a gradient step.
  5. Logs useful training statistics.
  6. Updates the target network periodically.

This keeps the core learning algorithm independent of any specific environment.

### 12.4. Agent Class (`agents/dqn_agent.py`)

This is the “fully assembled” agent used by your training script.

Responsibilities:

- Hold:
  - An instance of `DQNAlgorithm`.
  - The replay buffer.
  - Exploration schedule (\(\epsilon\)-greedy).
  - A reference to the environment.
- For each `train_episode()`:
  1. Reset the env, get initial state.
  2. For each step:
     - Select action using \(\epsilon\)-greedy from the **online network**.
     - Step the environment.
     - Store transition in replay buffer.
     - Call `dqn_algo.update(...)` once per environment step (after warmup).
  3. Aggregate episode stats (reward, steps, win/loss for Minesweeper).
  4. Log episode‑level stats.

This separation makes it easy to plug different algorithms into the same agent pattern.

---

## 13. TensorBoard Logging for DQN

Given your planned logging interface (via `logger.py` wrapper), here is what DQN should emit.

### 13.1. Episode-Level Metrics (Logged Once per Episode)

- `train/episode_reward`: total reward over the episode.
- `train/win_rate`: rolling win rate over the last 100 episodes (for Minesweeper, 1 if board cleared, 0 otherwise, averaged).
- `train/mine_density`: property of the environment (e.g. from `MinesweeperState`). For non‑Minesweeper envs you can skip or redefine this.
- `train/steps_per_ep`: number of steps taken in the episode.

These scalars let you see how learning is progressing across training.

### 13.2. Update-Level Metrics (Logged Every Gradient Update)

- `train/loss`: the DQN loss (mean squared TD error) on the current batch.
- `train/q_value_mean`: the mean of `Q(s,a; θ)` for the batch actions (a sanity check that Q-values stay in a reasonable range).
- `train/entropy`: mainly for policy‑based methods; for DQN you can either leave this `NaN`/0 or skip it.

### 13.3. Performance Metrics for Study Comparison

At least once at the end (or periodically), log:

- `perf/wall_time_per_ep`: average wall‑clock time per episode.
- `perf/inference_ms`: average time (in milliseconds) for one forward pass.
- `perf/gpu_memory_mb`: maximum or typical GPU memory usage in MB.

These logs allow you to create your planned **mine_density × win_rate** figure across board sizes, and overlay compute cost as bubble size.

---

## 14. Adapting DQN to Minesweeper

Although this document is general, here is how you can specialize DQN to your Minesweeper RL environment.

### 14.1. State Representation

Options:

- **Flat features**: e.g., integer grid encoded as channels: hidden cell, revealed number, flag, mine probability estimate, etc., then flattened into a vector.
- **Image‑like**: treat the board as a 2D grid with channels and use a small CNN.

Whatever you choose, it must be **fully observable**: the state passed to the Q-network should be everything the agent is allowed to see at that time.

### 14.2. Action Space

Define a discrete action space:

- E.g., `num_actions = board_height * board_width`, where each action means “reveal cell (i,j)”.
- Optionally, extra actions for “place flag” vs “reveal”, if your environment distinguishes them.

Map the discrete action index produced by DQN to the environment’s action object in your `env.step()`.

### 14.3. Reward Design

Basic scheme:

- Small positive reward for each safe reveal.
- Negative reward for hitting a mine (e.g., -1 or -10, ending the episode).
- Larger positive reward for fully clearing the board (win).

You can tweak reward scaling and shaping to make learning easier.

### 14.4. Logging Minesweeper-Specific Info

Besides generic DQN logs, make sure to log:

- `train/win_rate` per board size and mine density.
- `train/mine_density` from the environment.

Then you can post‑process `summary.csv` across algorithms to draw the **mine_density × win_rate** curve per board size (6×6, 8×8, 10×10, 16×16) with densities 0.10 → 0.25.

---

## 15. Checklist: Implementing DQN From Scratch

If you follow this checklist, you should be able to implement a working DQN agent.

1. **Environment ready**
   - Has `reset()`, `step(action)`, `observation_space`, `action_space`.
2. **Network**
   - Implement `DQNNetwork` mapping state → `num_actions` Q-values.
3. **Replay Buffer**
   - Implement `push` and `sample`.
4. **Algorithm class**
   - Maintains online and target networks.
   - Has `update()` that:
     - Samples batch.
     - Computes targets using target network.
     - Computes loss = mean squared TD error.
     - Optimizes network.
     - Updates target parameters every `target_update_freq` steps.
     - Logs `train/loss`, `train/q_value_mean`.
5. **Agent class**
   - Runs environment episodes.
   - Uses \(\epsilon\)-greedy on online network for action selection.
   - Stores transitions in replay buffer.
   - Calls `algorithm.update()` each step (after warmup).
   - Logs episode stats (`train/episode_reward`, `train/win_rate`, etc.).
6. **Training script**
   - Creates env, agent, logger.
   - Loops over episodes.
   - Saves models, TensorBoard logs, and summary CSV.

With these components in place and this document as a reference, you should be able to implement a clean, modular DQN agent for Minesweeper or any other discrete‑action RL environment.
