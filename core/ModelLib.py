import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List, Dict, Literal, Tuple, Union

observation_size = 4
action_space_size = 2
low_observation = [-2.4, -2.0, -0.42, -3.5]
high_observation = [2.4, 2.0, 0.42, 3.5]

ModelOutput = NamedTuple('ModelOutput', [('value', float), ('reward', float), (
    'policy', Dict[int, float]), ('encoded_state', List[float])])


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


class Model(nn.Module):
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
            encoded_state_size + action_space_size, dynamics_hidden_sizes, encoded_state_size + 1).to(device)

        self.eval()

    def encode_observation(self, observation: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if not torch.is_tensor(observation):
            observation = torch.from_numpy(
                observation.astype(np.float32)).to(self.device)
        encoded_state = self.representation_network(observation)
        return encoded_state

    def process_prediction(self, prediction: torch.Tensor, train: bool):
        policy, value = prediction[:, :-1], prediction[:, -1:]
        if train:
            policy_exp = policy.exp()
            policy_logits = policy_exp / policy_exp.sum()
            return policy_logits, value
        else:
            return policy.detach().numpy(), value.detach().numpy()

    def _initial_inference(self, observation: Union[torch.Tensor, np.ndarray], train: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_state = self.encode_observation(observation)
        prediction = self.prediction_network(encoded_state)
        policy, value = self.process_prediction(prediction, train)
        return policy, value, encoded_state

    def _recurrent_inference(self, encoded_state: torch.Tensor, action: Union[torch.Tensor, np.ndarray], train: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(action):
            action = torch.from_numpy(action.astype(np.float32))
        dynamics_input = torch.cat([encoded_state.T, action.T]).T
        dynamics = self.dynamics_network(dynamics_input)
        prediction = self.prediction_network(encoded_state)

        reward, next_encoded_state = dynamics[:, -1:], dynamics[:, :-1]
        policy, value = self.process_prediction(prediction, train)

        return reward, next_encoded_state, policy, value

    def initial_inference(self, observation: np.ndarray, train: bool = False) -> ModelOutput:
        if train:
            policy, value, encoded_state = self._initial_inference(
                observation, True)
        else:
            with torch.no_grad():
                policy, value, encoded_state = self._initial_inference(
                    observation, False)
        model_output = ModelOutput(value, 0, policy, encoded_state)
        return model_output

    def recurrent_inference(self, encoded_state: torch.Tensor, action: np.ndarray, train: bool = False) -> ModelOutput:
        if train:
            reward, next_encoded_state, policy, value = self._recurrent_inference(
                encoded_state, action, True)
        else:
            with torch.no_grad():
                reward, next_encoded_state, policy, value = self._recurrent_inference(
                    encoded_state, action, False)
                reward = reward.detach().numpy()

        model_output = ModelOutput(value, reward, policy, next_encoded_state)
        return model_output

    def predict_unroll(self, observations: Union[torch.Tensor, np.ndarray], actions: List[Union[torch.Tensor, np.ndarray]], train: bool = False):
        predictions = []
        model_output = self.initial_inference(observations, train=train)
        predictions.append(model_output)

        for action in actions:
            model_output = self.recurrent_inference(
                model_output.encoded_state, action, train=train)
            predictions.append(model_output)

        predictions = [(i.value, i.reward, i.policy) for i in predictions]
        return predictions


def update_weights(batch, model: Model, optimizer) -> Tuple[float, float, float]:
    model.train()
    value_loss, reward_loss, policy_loss = 0, 0, 0

    mse_loss = F.mse_loss
    bce_loss = nn.BCEWithLogitsLoss()

    observations, actions, targets = batch
    predictions = model.predict_unroll(observations, actions, train=True)
    for i in range(len(predictions)):
        values, rewards, polices = predictions[i]
        target_values, target_rewards, target_polices = targets[i]

        value_loss += mse_loss(values, target_values)
        policy_loss += bce_loss(polices, target_polices)
        if i > 0:
            reward_loss += mse_loss(rewards, target_rewards)
    total_loss = value_loss + reward_loss + policy_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    model.eval()

    return value_loss.item(), reward_loss.item(), policy_loss.item()
