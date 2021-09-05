from core import ModelLib
from core import GameLib
from core import utils
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
import datetime

seed = 42
lr = 0.001
device = 'cpu'
device = 'cuda'
encoded_state_size = 256
hidden_size = 128
hidden_layers = 5
num_unroll = 3
window_size = 5000
batch_size = 256
training_steps = 50
num_epochs = 15
discount = 0.997

env = gym.make('CartPole-v0')
# env = gym.make('CartPole-v1')

model_args = {
    'observation_size': env.observation_space.shape[0],
    'action_space_size': env.action_space.n,
    'encoded_state_size': encoded_state_size,
    'dynamics_hidden_sizes': [hidden_size] * hidden_layers,
    'prediction_hidden_sizes': [hidden_size] * hidden_layers,
    'representation_hidden_sizes': [hidden_size] * hidden_layers,
    'num_unroll': num_unroll,
    'lr': lr,
    'dropout': 0.075,
    # 'dropout': 0.1,
    'device': device,
}

env.seed(seed)
GameLib.np.random.seed(seed)
GameLib.torch.manual_seed(seed)
ModelLib.torch.manual_seed(seed)
utils.np.random.seed(seed)

model = ModelLib.MuModel(**model_args)
replay_buffer = GameLib.ReplayBuffer(
    window_size, batch_size, num_unroll, device)
optimizer = optim.Adam(model.parameters(), lr=lr)

returns = []
losses = []
start_time = datetime.datetime.now()
for training_step in range(training_steps):
    game = GameLib.play_game(env=env, model=model,
                             discount=discount, explore=True)
    replay_buffer.save_game(game)
    returns.append(game.game_return)
    average_returns = np.mean(returns[-10:])
    print('step %5u: , return: %3u, average return: %6.2f' %
          (training_step + 1, returns[-1], average_returns))

    for epoch in range(num_epochs):
        batch = replay_buffer.sample_vector(tensor=True)
        epoch_losses = ModelLib.update_weights(model, batch, optimizer)
        losses.append(epoch_losses)

print(datetime.datetime.now() - start_time)
losses = np.array(losses)

plt.plot(returns)
plt.savefig('returns.png')
plt.clf()
plt.yscale('log')
plt.plot(losses.sum(axis=1), label='total loss')
plt.plot(losses[:, 0], label='value loss')
# plt.plot(losses[:, 1], label='reward loss')
# plt.plot(losses[:, 2], label='policy loss')
# plt.legend()
plt.savefig('losses.png')
