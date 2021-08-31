import gym
import numpy as np
from collections import deque
import torch
from typing import List, Union


class Game:
    def __init__(self, env: gym.Env, discount: float) -> None:
        self.action_history: List[List[int]] = []
        self.rewards = []
        self.policies = []
        self.discount = discount

        self.env = env
        self.observations = [self.env.reset()]
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

    def step(self, action_vector: Union[List[float], np.ndarray]) -> None:
        observation, reward, done, _ = self.env.step(np.argmax(action_vector))
        self.observations += [observation]

        self.action_history.append(action_vector)
        self.rewards.append(reward)
        self.done = done

    def policy_step(self, policy: Union[List[float], np.ndarray], stochastic=True):
        if stochastic:
            idx = np.random.choice(np.arange(len(policy)), p=policy)
        else:
            idx = np.argmax(policy)
        action_vector = np.zeros(self.action_space_size)
        action_vector[idx] = 1

        self.step(action_vector)
        self.policies.append(policy)

    def make_image(self, state_index: int) -> np.ndarray:
        return self.observations[state_index].reshape(1, -1)

    def make_target(self, state_index: int, num_unroll: int):
        # no td bootstrapping
        targets = []
        for current_index in range(state_index, state_index + num_unroll + 1):
            if current_index < self.game_length:
                rewards = np.array(self.rewards[current_index:])
                value = (rewards * (self.discount **
                         np.arange(len(rewards)))).sum()
                target = (value, rewards[0], self.policies[current_index])
            else:
                target = (0, 0, np.ones(self.action_space_size) /
                          self.action_space_size)
            targets.append(target)
        return targets


class ReplayBuffer:

    def __init__(self, window_size: int, batch_size: int, num_unroll: int, action_space_size: int) -> None:
        self.window_size, self.batch_size, self.num_unroll = window_size, batch_size, num_unroll
        self.action_space_size = action_space_size
        self.buffer: List[Game] = deque(maxlen=self.window_size)

    def save_game(self, game: Game) -> None:
        self.buffer.append(game)

    def sample_game(self) -> Game:
        return np.random.choice(self.buffer)

    def sample_position(self, game: Game) -> int:
        return np.random.randint(0, game.game_length - 1)

    def sample_batch(self):
        def extend_actions(action_history):
            action_history = action_history[:self.num_unroll]
            short = self.num_unroll - len(action_history)
            if short > 0:
                action_history += [list(np.ones(self.action_space_size) /
                                        self.action_space_size)] * short
            return action_history

        games = [self.sample_game() for i in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]

        return [(
            g.make_image(i),
            extend_actions(g.action_history[i:]),
            g.make_target(i, self.num_unroll)
        ) for(g, i) in game_pos
        ]

    def sample_vector(self, tensor: bool = False):
        def to_array(x):
            if tensor:
                return torch.tensor(x, dtype=torch.float32).reshape(self.batch_size, -1)
            else:
                return np.array(x).reshape(self.batch_size, -1)

        batch = self.sample_batch()
        batch_obvservations, batch_actions, batch_targets = zip(*batch)

        batch_obvservations = to_array(batch_obvservations)
        batch_actions = [to_array([i[j] for i in batch_actions])
                         for j in range(self.num_unroll)]
        batch_values = [to_array([i[j][0] for i in batch_targets])
                        for j in range(self.num_unroll + 1)]
        batch_rewards = [to_array([i[j][1] for i in batch_targets])
                         for j in range(self.num_unroll + 1)]
        batch_policy = [to_array([i[j][2] for i in batch_targets])
                        for j in range(self.num_unroll + 1)]

        batch_targets = list(zip(batch_values, batch_rewards, batch_policy))
        return batch_obvservations, batch_actions, batch_targets
