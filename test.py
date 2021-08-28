from core.model import MuZeroConfig, SharedStorage
from core.utils import ReplayBuffer, selfplay, KnownBounds, train_network
from cartpole import CartPole, CartPoleNetwork
import torch.optim as optim


def visit_softmax_temperature(num_moves, training_steps):
    return 1


device = 'cpu'
action_space_size = 2
hidden_state_size = 128

config = MuZeroConfig(
    game_class=CartPole,
    action_space_size=action_space_size,
    num_training_loops=50,
    num_episodes=50,
    num_epochs=50,
    max_moves=1000,
    discount=0.99,
    dirichlet_alpha=0.25,
    num_simulations=11,
    batch_size=512,
    td_steps=10,
    num_actors=10,
    lr_init=0.05,
    lr_decay_steps=1,
    visit_softmax_temperature_fn=visit_softmax_temperature,
    known_bounds=KnownBounds(-1, 1),
    device=device,
)

network_args = {
    'encoded_state_size': 128,
    'dynamics_hidden_sizes': [128] * 4,
    'prediction_hidden_sizes': [128] * 4,
    'representation_hidden_sizes': [128] * 4,
    'device': device,
}

# network = CartPoleNetwork(
#     encoded_state_size=128,
#     dynamics_hidden_sizes=[128] * 4,
#     prediction_hidden_sizes=[128] * 4,
#     representation_hidden_sizes=[128] * 4,
#     device=device,
# )


storage = SharedStorage(config, CartPoleNetwork, network_args)
replay_buffer = ReplayBuffer(config)
optimizer = optim.Adam
for i in range(config.num_training_loops):
    selfplay_return = selfplay(config, storage, replay_buffer)
    network = train_network(config, storage, replay_buffer, optimizer)
    storage.save_network(i, network)

    print('training loop %6u:\tselfplay return: %6.1f' %
          (i + 1, selfplay_return))

# if i == 1:
#     quit()
# # train_network(config, storage, replay_buffer)
# # return storage.latest_network()

# network = storage.new_network()
# print(network)
