import sys
import os
import numpy as np
import torch
from TicTacToe import TicTacToe
from Agent import Agent
import tqdm

def main():
    np.random.seed(3)
    torch.manual_seed(1)

    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")

    player1 = Agent(epsilon=0.5, learning_rate=0.003, player_id=1)
    player2 = Agent(epsilon=0.5, learning_rate=0.003, player_id=2)

    USE_TRAINED_MODELS = True  # Set to True to use pre-trained models

    if USE_TRAINED_MODELS:
        checkpoint_data1 = torch.load('checkpoint/tictactoe_player1_model.pt')
        player1.qmodel.load_state_dict(checkpoint_data1['state_dict'])
        player1.optimizer.load_state_dict(checkpoint_data1['optimizer'])
        checkpoint_data2 = torch.load('checkpoint/tictactoe_player2_model.pt')
        player2.qmodel.load_state_dict(checkpoint_data2['state_dict'])
        player2.optimizer.load_state_dict(checkpoint_data2['optimizer'])
        player1.epsilon = 0.0  # No random guesses
        player2.epsilon = 0.0  # No random guesses

    game = TicTacToe(player1, player2)

    if USE_TRAINED_MODELS:
        # Test the trained model
        _, won_count, draw_count, lost_count, total_count = game.play(num_games=100, visualize=True)
        print(f"Percentage won = {won_count}, Percentage lost = {lost_count}, Percentage draw = {draw_count}, Total count = {total_count}")
        return

    # Training code
    total_number_of_games = 1000000
    number_of_games_per_batch = 200
    min_loss = np.inf

    for i in tqdm.trange(total_number_of_games // number_of_games_per_batch):
        transitions1, transitions2, _, _, _, _ = game.play(num_games=number_of_games_per_batch)
        np.random.shuffle(transitions1)
        np.random.shuffle(transitions2)
        loss1 = player1.do_Qlearning_on_agent_model(transitions1)
        loss2 = player2.do_Qlearning_on_agent_model(transitions2)

        if loss1 < min_loss and loss1 < 0.01:
            min_loss = loss1

        tqdm.tqdm.write(f"Loss: {loss1}, Min Loss: {min_loss}")

        # Save models
        checkpoint_data1 = {
            'state_dict': player1.qmodel.state_dict(),
            'optimizer': player1.optimizer.state_dict()
        }
        ckpt_path1 = os.path.join("checkpoint/tictactoe_player1_model.pt")
        torch.save(checkpoint_data1, ckpt_path1)

        checkpoint_data2 = {
            'state_dict': player2.qmodel.state_dict(),
            'optimizer': player2.optimizer.state_dict()
        }
        ckpt_path2 = os.path.join("checkpoint/tictactoe_player2_model.pt")
        torch.save(checkpoint_data2, ckpt_path2)

    # Enhancement: Fine-tune on losing cases
    losing_transitions = []
    for _ in range(100):  # Play 100 games to collect losing cases
        transitions1, _, _, _, lost_count, _ = game.play(num_games=1)
        if lost_count > 0:
            losing_transitions.extend(transitions1)

    # Fine-tune the model on losing cases
    for _ in range(100):  # Fine-tune for 100 epochs
        np.random.shuffle(losing_transitions)
        player1.do_Qlearning_on_agent_model(losing_transitions)

    # Save the fine-tuned model
    checkpoint_data1 = {
        'state_dict': player1.qmodel.state_dict(),
        'optimizer': player1.optimizer.state_dict()
    }
    ckpt_path1 = os.path.join("checkpoint/tictactoe_player1_model_finetuned.pt")
    torch.save(checkpoint_data1, ckpt_path1)

if __name__ == "__main__":
    sys.exit(int(main() or 0))