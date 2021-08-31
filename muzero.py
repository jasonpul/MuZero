from core import ModelLib
from core import GameLib
from core import utils
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt

device = 'cpu'
discount = 0.997
window_size = 5000
batch_size = 256
num_unroll = 3
num_epochs = 20
training_steps = 100
lr = 0.001
seed = 42

model_args = {
    'encoded_state_size': 128,
    'dynamics_hidden_sizes': [128] * 4,
    'prediction_hidden_sizes': [128] * 4,
    'representation_hidden_sizes': [128] * 4,
    'device': device,
}

env = gym.make('CartPole-v1')
action_space_size = env.action_space.n

env.seed(seed)
np.random.seed(seed)
ModelLib.np.random.seed(seed)
ModelLib.torch.manual_seed(seed)
GameLib.np.random.seed(seed)
GameLib.torch.manual_seed(seed)
utils.np.random.seed(seed)

replay_buffer = GameLib.ReplayBuffer(window_size, batch_size, num_unroll, 2)
model = ModelLib.Model(**model_args)
optimizer = optim.Adam(model.parameters(), lr=lr)

returns = []
losses = []
for training_step in range(training_steps):
    game = utils.play_game(env, model, action_space_size, num_unroll, discount)
    replay_buffer.save_game(game)
    step_return = game.game_return
    returns.append(step_return)
    print(step_return)

    for epoch in range(num_epochs):
        batch = replay_buffer.sample_vector(tensor=True)
        epoch_losses = ModelLib.update_weights(batch, model, optimizer)
        losses.append(epoch_losses)

losses = np.array(losses)

plt.plot(returns)
plt.savefig('returns.png')
plt.clf()
plt.plot(losses[:, 0])
plt.plot(losses[:, 1])
plt.plot(losses[:, 2])
plt.savefig('losses.png')
