from __future__ import annotations
from .utils import naive_search
import numpy as np
import torch
from collections import deque
from typing import List, Union, Literal, TYPE_CHECKING, Tuple, NamedTuple, Dict
if TYPE_CHECKING:
    import gym
    import numpy as np
    from .ModelLib import MuModel

Target = NamedTuple('Target', [('value', float), ('reward', float), (
    'policy', Dict[int, float])])


class Game():
    def __init__(self, env: gym.Env, discount: float = 0.95) -> None:
        self.action_history = []
        self.rewards = []
        self.policies = []
        self.discount = discount

        self.env = env
        self.observations = [env.reset()]
        self.done = False
        self.action_space_size = self.env.action_space.n

    @property
    def terminal(self) -> bool:
        return self.done

    @property
    def game_return(self) -> float:
        return sum(self.rewards)

    @property
    def game_length(self) -> int:
        return len(self.action_history)

    def step(self, action_index: int) -> None:
        observation, reward, done, _ = self.env.step(action_index)

        self.observations.append(observation)
        self.rewards.append(reward)
        self.done = done

    def policy_step(self, policy: Union[List[float], np.ndarray], stochastic: bool = True) -> None:
        if stochastic:
            idx = np.random.choice(list(range(len(policy))), p=policy)
        else:
            idx = np.argmax(policy)
        action_vector = np.zeros(self.action_space_size)
        action_vector[idx] = 1

        self.step(idx)
        self.policies.append(policy)
        self.action_history.append(action_vector)

    def make_image(self, index: int) -> np.ndarray:
        return self.observations[index].reshape(1, -1)

    def make_target(self, state_index: int, num_unroll: int) -> List[Target]:
        targets = []
        for current_index in range(state_index, state_index + num_unroll + 1):
            rewards = np.array(self.rewards[current_index:])
            value = (rewards * (self.discount ** np.arange(len(rewards)))).sum()

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
                # print(current_index, len(self.rewards))
                # last_reward = self.rewards[current_index]
            else:
                last_reward = 0

            if current_index < len(self.policies):
                targets.append(Target(value, last_reward,
                               self.policies[current_index]))
            else:
                targets.append(
                    Target(0, last_reward, np.zeros(self.action_space_size)))
        return targets

    def make_target2(self, state_index: int, num_unroll: int) -> List[Target]:
        targets = []
        for current_index in range(state_index, state_index + num_unroll + 1):
            rewards = np.array(self.rewards[current_index:])
            value = (rewards * (self.discount ** np.arange(len(rewards)))).sum()

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
                # print(current_index, len(self.rewards))
                # last_reward = self.rewards[current_index]
            else:
                last_reward = 0

            if current_index < len(self.policies):
                targets.append(Target(value, last_reward,
                               self.policies[current_index]))
            else:
                targets.append(
                    Target(0, last_reward, np.zeros(self.action_space_size)))
        return targets


class ReplayBuffer():
    def __init__(self, window_size: int, batch_size: int, num_unroll: int, device: Literal['cpu', 'cuda']) -> None:
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_unroll = num_unroll
        self.buffer: List[Game] = deque(maxlen=window_size)
        self.device = device

    def save_game(self, game: Game) -> None:
        self.buffer.append(game)

    def sample_game(self) -> Game:
        return np.random.choice(self.buffer)

    def sample_position(self, game: Game) -> int:
        return np.random.randint(0, game.game_length - 1)

    def sample_batch(self) -> List[Tuple[np.ndarray, List[int], List[Target]]]:
        def extend_actions(actions):
            actions = actions[:self.num_unroll]
            short = self.num_unroll - len(actions)
            if short > 0:
                actions += [np.zeros(actions[0].shape)] * short
            return actions

        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]

        return [(
            g.make_image(i),
            extend_actions(g.action_history[i:]),
            g.make_target(i, self.num_unroll)
        ) for (g, i) in game_pos]

    def sample_vector(self, tensor: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple(torch.Tensor, torch.Tensor, torch.Tensor)]]:
        def to_array(x):
            if tensor:
                return torch.tensor(x, dtype=torch.float32).reshape(self.batch_size, -1).to(self.device)
            else:
                return np.array(x).reshape(self.batch_size, -1)

        batch = self.sample_batch()
        batch_obvservations, batch_actions, batch_targets = zip(*batch)

        batch_obvservations = to_array(batch_obvservations)
        batch_actions = [to_array([i[j] for i in batch_actions])
                         for j in range(self.num_unroll)]
        batch_values = [to_array([target[i].value for target in batch_targets])
                        for i in range(self.num_unroll + 1)]
        batch_rewards = [to_array([target[i].reward for target in batch_targets])
                         for i in range(self.num_unroll + 1)]
        batch_policy = [to_array([target[i].policy for target in batch_targets])
                        for i in range(self.num_unroll + 1)]

        batch_targets = list(zip(batch_values, batch_rewards, batch_policy))
        return batch_obvservations, batch_actions, batch_targets


def play_game(env: gym.Env, model: MuModel, discount: float = 0.997, explore: bool = True):
    game = Game(env, discount=discount)
    while not game.terminal:
        observation = game.make_image(-1)
        epsilon = np.random.random()
        if (epsilon < 0.05) and explore:
            policy = [1 / model.action_space_size] * model.action_space_size
        else:
            policy = naive_search(model, observation, temperature=1)
        game.policy_step(policy, stochastic=True)
    return game
