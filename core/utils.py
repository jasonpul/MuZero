from __future__ import annotations
import numpy as np
import torch
import itertools
from typing import List, TYPE_CHECKING, Union
if TYPE_CHECKING:
    from .ModelLib import MuModel


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def to_one_hot(x: Union[List, np.ndarray], dim: int) -> np.ndarray:
    one_hot = np.zeros((len(x), dim))
    one_hot[np.arange(len(x)), x] = 1
    return one_hot


def get_action_permutations(action_space_size: int, num_unroll: int) -> List[np.ndarray]:
    action_permutations = list(itertools.product(
        list(range(action_space_size)), repeat=num_unroll))
    action_history = [to_one_hot(
        [i[j] for i in action_permutations], action_space_size) for j in range(num_unroll)]

    return action_history


def naive_search(model: MuModel, observation: np.ndarray, temperature: float = 1) -> np.ndarray:
    num_unroll, action_space_size = model.num_unroll, model.action_space_size
    num_permutations = action_space_size ** num_unroll
    actions = get_action_permutations(action_space_size, num_unroll)
    oberservations = np.repeat(observation, num_permutations, axis=0)

    oberservations = torch.from_numpy(
        oberservations.astype(np.float32)).to(model.device)
    actions = [torch.from_numpy(a.astype(np.float32)).to(model.device)
               for a in actions]

    prediction = model.predict_unroll(oberservations, actions)
    values = prediction[-1].value

    minimum = min(values)
    maximum = max(values)
    values = (values - minimum) / (maximum - minimum)

    policy = np.zeros(action_space_size)
    for i in range(num_permutations):
        idx = np.argmax(actions[0][i].cpu().numpy())
        policy[idx] += values[i, 0]
    policy = softmax(policy / temperature)
    return policy


def mcts():
    pass
