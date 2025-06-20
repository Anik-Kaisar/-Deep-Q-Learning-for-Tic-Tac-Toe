🤖 Deep Q-Learning for Tic-Tac-Toe

This project demonstrates the use of **Deep Q-Learning** to train intelligent agents to play **Tic-Tac-Toe** competitively. Two agents, each powered by their own neural networks, learn to play through repeated self-play and improve using experience replay and Q-learning.

🎯 Project Highlights

* Implements **two independent learning agents**
* Encodes game state using **3x10 one-hot encoded tensors**
* Learns optimal actions for all possible Tic-Tac-Toe board configurations
* Tracks wins, losses, draws, and learns over time through feedback
* Models saved and reused to evaluate agent performance post-training

---

🧱 Project Structure

```
TicTacToeDQN/
├── QModel.py              # Deep Q-network using PyTorch
├── Agent.py               # Agent logic: Q-policy, epsilon-greedy, and learning
├── TicTacToe.py           # Environment for simulating the game
├── TicTacToeDQN_main.py   # Training and evaluation pipeline
├── checkpoint/            # Saved models for player 1 and 2
```

---
⚙️ Technical Features

* Q-network includes:

  * **Embedding Layer** for state input (cells & turn)
  * **Multi-layer MLP** with ReLU activations
  * **Softmax output** for selecting one of 9 possible moves
* Learning strategy:

  * **ε-greedy** action selection during training
  * **Reward shaping** for win/loss/tie outcomes
  * **Smooth L1 loss** for stable Q-value updates
* Evaluation:

  * After training, models are evaluated on 100+ games
  * Reports win/loss/draw percentage

---

🧠 Concepts Demonstrated

* Reinforcement Learning (Q-Learning)
* Deep Neural Networks for Q(s, a)
* Self-play Learning Loop
* Game State Encoding (Tensor Representation)
* PyTorch for Network Training

---

