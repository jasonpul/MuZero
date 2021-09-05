from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, NamedTuple, Dict, Literal

ModelOutput = NamedTuple('ModelOutput', [('value', float), ('reward', float), (
    'policy', Dict[int, float]), ('encoded_state', List[float])])


def scale_encoded_state(state: torch.Tensor):
    # min_state = state.amin(dim=1, keepdim=True)
    # max_state = state.amax(dim=1, keepdim=True)
    # normalized_state = (state - min_state) / (max_state - min_state)
    # return normalized_state
    return state


class DenseNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], outputs: List[int], hidden_activation=nn.ELU, output_activation=nn.Identity, dropout: Optional[float] = None) -> None:
        super().__init__()
        dropout = dropout if dropout is not None else 0

        layers = [nn.Linear(input_size, hidden_sizes[0]), hidden_activation()]
        for i in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[i],
                                 hidden_sizes[i + 1]), hidden_activation(), nn.Dropout(p=dropout)]

        self.linear_stack = nn.Sequential(*layers)
        self.output_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_sizes[-1], o), output_activation()) for o in outputs])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear_stack(x)
        out = [layer(out) for layer in self.output_layers]
        if len(out) == 1:
            return out[0]
        return out


class MuModel(nn.Module):
    def __init__(self, observation_size: int, action_space_size: int, encoded_state_size: int, dynamics_hidden_sizes: List[int], prediction_hidden_sizes: List[int], representation_hidden_sizes: List[int], num_unroll: int, lr: float, dropout: Optional[float] = None, device: Literal['cpu', 'cuda'] = 'cpu') -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_space_size = action_space_size
        self.encoded_state_size = encoded_state_size
        self.num_unroll = num_unroll
        self.lr = lr
        self.device = device

        self.representation_network = DenseNetwork(
            input_size=observation_size,
            hidden_sizes=representation_hidden_sizes,
            outputs=[encoded_state_size],
            dropout=dropout
        )
        self.dynamics_network = DenseNetwork(
            input_size=encoded_state_size + action_space_size,
            hidden_sizes=dynamics_hidden_sizes,
            outputs=[1, encoded_state_size],

            dropout=dropout
        )
        self.prediction_network = DenseNetwork(
            input_size=encoded_state_size,
            hidden_sizes=prediction_hidden_sizes,
            outputs=[action_space_size, 1],

            dropout=dropout
        )

        self.to(device)
        self.eval()

    def encode_state(self, observation: torch.Tensor) -> torch.Tensor:
        encoded_state = self.representation_network(observation)
        scaled_encoded_state = scale_encoded_state(encoded_state)
        return scaled_encoded_state

    def initial_inference(self, observation: torch.Tensor) -> ModelOutput:
        encoded_state = self.encode_state(observation)
        policy, value = self.prediction_network(encoded_state)
        model_output = ModelOutput(value, 0, policy, encoded_state)
        return model_output

    def recurrent_inference(self, encoded_state: torch.Tensor, action: torch.Tensor) -> ModelOutput:
        dynamics_input = torch.cat([encoded_state.T, action.T]).T
        reward, next_encoded_state = self.dynamics_network(dynamics_input)
        policy, value = self.prediction_network(encoded_state)
        model_output = ModelOutput(value, reward, policy, next_encoded_state)
        return model_output

    def _predict_unroll(self, observation: torch.Tensor, actions: List[torch.Tensor]) -> List[ModelOutput]:
        predictions = []
        model_output = self.initial_inference(observation)
        predictions.append(model_output)

        for k in range(self.num_unroll):
            model_output = self.recurrent_inference(
                model_output.encoded_state, actions[k])
            predictions.append(model_output)

        return predictions

    def predict_unroll(self, observation: torch.Tensor, actions: List[torch.Tensor], train: bool = False) -> List[ModelOutput]:

        if train:
            predictions = self._predict_unroll(observation, actions)
        else:
            with torch.no_grad():
                predictions = self._predict_unroll(observation, actions)
        return predictions


def update_weights(model: MuModel, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], optimizer) -> Tuple[float, float, float]:
    model.train()
    obvservations, actions, targets = batch

    mse, smcel = F.mse_loss, nn.BCEWithLogitsLoss()
    predictions = model.predict_unroll(obvservations, actions, train=True)

    value_loss, reward_loss, policy_loss = 0, 0, 0
    for i in range(len(predictions)):
        target_values, target_rewards, target_policies = targets[i]

        value_loss += mse(predictions[i].value, target_values)
        policy_loss += smcel(predictions[i].policy, target_policies)
        if i > 0:
            reward_loss += mse(predictions[i].reward, target_rewards)

    total_loss = value_loss + reward_loss + policy_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    model.eval()
    return value_loss.item(), reward_loss.item(), policy_loss.item()
