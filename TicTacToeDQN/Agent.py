import torch
import numpy as np
from QModel import QModel

class Agent:
    def __init__(self, epsilon=0.2, learning_rate=0.01, gamma=0.9, player_id=1):
        self.player_id = player_id
        self.qmodel = QModel()
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.qmodel.parameters(), lr=learning_rate)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # for epsilon-greedy during training

    def get_random_action(self):
        return np.random.randint(0, 9)

    def get_Q_action(self, state):
        with torch.no_grad():
            state2d, turn = state
            turns = torch.tensor(turn, dtype=torch.int64)[None]
            mask = ~(state2d != 0) * 1  # empty cells will be true
            mask2 = (state2d != 0) * -1000
            states2d = torch.tensor(state2d, dtype=torch.int64)[None]
            qvalues = self.qmodel(states2d, turns)[0]
            masked_qactions = qvalues * mask.flatten() + mask2.flatten()
            action = np.argmax(masked_qactions)
            ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
            while state[0][ax, ay] != 0:  # invalid move
                action = self.get_random_action()
                ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
            return action

    def get_epsilon_greedy_action(self, state):
        if np.random.rand() < self.epsilon:
            action = self.get_random_action()
            ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
            while state[0][ax, ay] != 0:  # cell is occupied
                action = self.get_random_action()
                ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
        else:
            action = self.get_Q_action(state)
            ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
            while state[0][ax, ay] != 0:  # invalid move
                action = self.get_random_action()
                ax, ay = torch.div(action, 3, rounding_mode='trunc'), action % 3
        return action

    def do_Qlearning_on_agent_model(self, state_action_nstate_rewards):
        states, actions, next_states, rewards = zip(*state_action_nstate_rewards)
        states2d, turns = zip(*states)
        next_states2d, next_turns = zip(*next_states)
        turns = torch.tensor(turns, dtype=torch.int64)
        next_turns = torch.tensor(next_turns, dtype=torch.int64)
        states2d = torch.tensor(states2d, dtype=torch.int64)
        next_states2d = torch.tensor(next_states2d, dtype=torch.int64)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        with torch.no_grad():
            mask = (next_turns > 0).float()  # whether the game is over or not
            next_qvalues = self.qmodel(next_states2d, next_turns)
            expected_qvalues_for_actions = rewards + (self.gamma * torch.max(next_qvalues, 1)[0])

        qvalues_for_actions = torch.gather(self.qmodel(states2d, turns), dim=1, index=actions[:, None]).view(-1)
        loss = torch.nn.functional.smooth_l1_loss(qvalues_for_actions, expected_qvalues_for_actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()