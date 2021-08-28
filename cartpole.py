from core.model import AbstractNetwork, DenseNetwork, NetworkOutput
from core.utils import Action, AbstractGame
from core.gym_wrappers import ScalingObservationWrapper
import numpy as np
import math
import torch
import gym

from typing import List, Literal

observation_size = 4
action_space_size = 2
low_observation = [-2.4, -2.0, -0.42, -3.5]
high_observation = [2.4, 2.0, 0.42, 3.5]


class CartPole(AbstractGame):

    def __init__(self, discount: float):
        super().__init__(discount)
        self.env = gym.make('CartPole-v1')
        self.env = ScalingObservationWrapper(
            self.env, low=low_observation, high=high_observation)
        self.actions = [Action(i) for i in range(self.env.action_space.n)]
        self.observations = [self.env.reset()]
        self.done = False

    @property
    def action_space_size(self) -> int:
        return len(self.actions)

    def step(self, action) -> int:
        observation, reward, done, _ = self.env.step(action.index)
        self.observations += [observation]
        self.done = done
        return reward

    @property
    def terminal(self) -> bool:
        return self.done

    def legal_actions(self) -> List[Action]:
        return self.actions

    def make_image(self, state_index: int):
        return self.observations[state_index]


class CartPoleNetwork(AbstractNetwork, torch.nn.Module):
    def __init__(self, encoded_state_size: int, dynamics_hidden_sizes: List[int], prediction_hidden_sizes: List[int], representation_hidden_sizes: List[int], device: Literal['cpu', 'cuda']) -> None:
        super().__init__()

        self.encoded_state_size = encoded_state_size
        self.dynamics_hidden_sizes = dynamics_hidden_sizes
        self.prediction_hidden_sizes = prediction_hidden_sizes
        self.representation_hidden_sizes = representation_hidden_sizes
        self.device = device

        self.representation_network = DenseNetwork(
            observation_size, representation_hidden_sizes, encoded_state_size).to(device)
        self.prediction_network = DenseNetwork(
            encoded_state_size, prediction_hidden_sizes, action_space_size + 1).to(device)
        self.dynamics_network = DenseNetwork(
            encoded_state_size + 1, dynamics_hidden_sizes, encoded_state_size + 1).to(device)

        self.eval()

    def encode_observation(self, observation: np.ndarray):
        observation = torch.from_numpy(
            observation.astype(np.float32)).to(self.device)
        encoded_state = self.representation_network(observation)
        return encoded_state

    def _initial_inference(self, observation: np.ndarray):
        encoded_state = self.encode_observation(observation)
        prediction_output = self.prediction_network(encoded_state)
        return prediction_output, encoded_state

    def _recurrent_inference(self, encoded_state, action):
        a = torch.tensor(action.index, dtype=torch.float32,
                         device=self.device).view(1, 1)
        dynamics_input = torch.cat(
            [encoded_state.view(self.encoded_state_size, 1), a]).squeeze()

        dynamics_output = self.dynamics_network(dynamics_input)
        prediction_output = self.prediction_network(encoded_state)
        return dynamics_output, prediction_output

    def initial_inference(self, observation: np.ndarray, train=False) -> NetworkOutput:
        if train:
            prediction_output, encoded_state = self._initial_inference(
                observation)
            policy = prediction_output[:-1]
            value = prediction_output[-1]

            policy_exp = policy.exp()
            policy_logits = policy_exp / policy_exp.sum()
        else:
            with torch.no_grad():
                prediction_output, encoded_state = self._initial_inference(
                    observation)
                policy = prediction_output[:-1]
                value = prediction_output[-1].item()

                policy_logits = {Action(i): policy[i].item()
                                 for i in range(action_space_size)}

        network_output = NetworkOutput(value, 0, policy_logits, encoded_state)
        return network_output

    def recurrent_inference(self, encoded_state: torch.Tensor, action: Action, train=False) -> NetworkOutput:
        if train:
            dynamics_output, prediction_output = self._recurrent_inference(
                encoded_state, action)
            reward = dynamics_output[-1]
            policy = prediction_output[:-1]
            value = prediction_output[-1]

            policy_exp = policy.exp()
            policy_logits = policy_exp / policy_exp.sum()

            # policy_logits = policy.exp() / policy.sum()
            # print(policy, policy.exp())
            # print(policy, policy_logits)
            # print(math.isnan(policy[0].item()))
        else:
            with torch.no_grad():
                dynamics_output, prediction_output = self._recurrent_inference(
                    encoded_state, action)
                reward = dynamics_output[-1].item()
                policy = prediction_output[:-1]
                value = prediction_output[-1].item()

                policy_logits = {Action(i): policy[i].item()
                                 for i in range(action_space_size)}

        next_encoded_state = dynamics_output[:-1]
        network_output = NetworkOutput(
            value, reward, policy_logits, next_encoded_state)
        return network_output

    # def train(self):
    #     self.representation_network.train()
    #     self.prediction_network.train()
    #     self.dynamics_network.train()

    # def eval(self):
    #     self.representation_network.eval()
    #     self.prediction_network.eval()
    #     self.dynamics_network.eval()
