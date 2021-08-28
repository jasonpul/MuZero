from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from .utils import Action, KnownBounds, AbstractGame
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Literal, Tuple, Optional, NamedTuple


class MuZeroConfig:

    def __init__(self, game_class: AbstractGame, action_space_size: int, num_training_loops: int, num_episodes: int, num_epochs: int, max_moves: int, discount: float, dirichlet_alpha: float, num_simulations: int, batch_size: int, td_steps: int, num_actors: int, lr_init: float, lr_decay_steps: float, visit_softmax_temperature_fn, known_bounds: Optional[KnownBounds] = None, device: Literal['cpu', 'cuda'] = 'cpu'):
        self.game_class = game_class

        # Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors
        self.num_training_loops = num_training_loops
        self.num_episodes = num_episodes
        self.num_epochs = num_epochs

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        # Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = int(1e3)
        self.window_size = int(1e6)
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        self.weight_decay = 1e-4
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = lr_init
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = lr_decay_steps

        self.device = device

    def new_game(self) -> AbstractGame:
        return self.game_class(self.discount)


NetworkOutput = NamedTuple('NetworkOutput', [('value', float), ('reward', float), (
    'policy_logits', Dict[Action, float]), ('encoded_state', List[float])])


class AbstractNetwork(ABC):
    steps = 0

    def __init__(self) -> None:
        super().__init__()
        self._training_steps = 0

    def training_steps(self) -> int:
        return self._training_steps

    @abstractmethod
    def initial_inference(self, observation) -> NetworkOutput:
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action) -> NetworkOutput:
        pass


class DenseNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, hidden_activation=nn.ELU, output_activation=nn.Identity) -> None:
        super().__init__()

        layers = [nn.Linear(input_size, hidden_sizes[0]), hidden_activation()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i],
                                 hidden_sizes[i + 1]), hidden_activation()]
        layers += [nn.Linear(hidden_sizes[-1], output_size),
                   output_activation()]
        self.linear_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear_stack(x)
        return out


class SharedStorage:

    def __init__(self, config: MuZeroConfig, network: AbstractNetwork, network_args):
        self.network_args = network_args
        self._network = network
        self._networks = {}
        self.config = config

    def new_network(self) -> AbstractNetwork:
        return self._network(**self.network_args)

    def latest_network(self) -> AbstractNetwork:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            # return UniformNetwork(self.config.action_space_size)
            return self.new_network()

    def save_network(self, step: int, network: AbstractNetwork):
        self._networks[step] = network
