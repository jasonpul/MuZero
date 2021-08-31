from .ModelLib import Model, ModelOutput
from .GameLib import Game
import math
import numpy as np
import itertools

from typing import List, Dict, Optional, Tuple, NamedTuple, Union

_pb_c_base = 19652
_pb_c_init = 1.25

KnownBounds = NamedTuple('KnownBounds', [('min', float), ('max', float)])


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=0)


class Node:
    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children: Dict[int, Node] = {}
        self.encoded_state = None
        self.reward = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MinMaxStats:
    def __init__(self, known_bounds: Optional[KnownBounds] = None) -> None:
        if known_bounds is not None:
            self.maximum = known_bounds.max
            self.minimum = known_bounds.min
            self.inf = False
        else:
            self.maximum = float('inf')
            self.minimum = -float('inf')
            self.inf = True

    def update(self, value: float):
        if not self.inf:
            self.maximum = max(self.maximum, value)
            self.minimum = max(self.minimum, value)

    def normalize(self, value: float) -> float:
        if (self.maximum > self.minimum) and not self.inf:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats, pb_c_base: Optional[float] = None, pb_c_init: Optional[float] = None) -> float:
    pb_c_base = _pb_c_base if pb_c_base is None else pb_c_base
    pb_c_init = _pb_c_init if pb_c_init is None else pb_c_init

    pb_c = math.log((parent.visit_count + pb_c_base + 1) /
                    pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value)
    return prior_score + value_score


def select_child(node: Node, min_max_stats: MinMaxStats) -> Tuple[int, Node]:
    _, action, child = max((ucb_score(node, child, min_max_stats), action, child)
                           for action, child, in node.children.items())
    return action, child


def expand_node(node: Node, model_output: ModelOutput):
    node.encoded_state = model_output.encoded_state
    node.reward = model_output.reward
    for action, logit in enumerate(model_output.policy):
        node.children[action] = Node(logit)


def backpropagate(search_path: List[Node], value: float, discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value)

        value = node.reward + discount * value


def run_mcts(root: Node, action_history: List[int], model: Model, num_simulations: int, known_bounds: KnownBounds = KnownBounds(0, 0), discount: float = 0.99):
    min_max_stats = MinMaxStats(known_bounds)

    for _ in range(num_simulations):
        history = list(action_history)
        node = root
        search_path = [node]

        while not node.is_leaf:
            action, node = select_child(node, min_max_stats)
            history.append(action)
            search_path.append(node)

        parent = search_path[-2]
        model_output = model.recurrent_inference(parent.encoded_state, action)

        expand_node(node, model_output)
        backpropagate(search_path, model_output.value, discount, min_max_stats)


def softmax_sample(visits: List[Tuple[int, int]], t: float) -> Tuple[int, List[float]]:
    visit_counts, actions = zip(*visits)
    visit_counts_exp = np.exp(visit_counts) * (1 / t)
    probs = visit_counts_exp / visit_counts_exp.sum()

    return np.random.choice(actions, p=probs), probs


def select_action(node: Node, temperature: float) -> Tuple[int, List[float]]:
    visits = [(child.visit_count, action)
              for action, child in node.children.items()]

    action, policy = softmax_sample(visits, temperature)
    return action, policy


def add_exploration_noise(node: Node, dirichlet_alpha: float, root_exploration_fraction: float):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    frac = root_exploration_fraction

    for a, n, in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


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


def naive_search(model: Model, observation: np.ndarray, action_space_size: float, num_unroll: int, temperature: float):
    action_histories = get_action_permutations(action_space_size, num_unroll)
    num_permutations = action_space_size ** num_unroll
    observations = np.repeat(observation, num_permutations, axis=0)
    prediction = model.predict_unroll(observations, action_histories)
    values = prediction[-1][0]

    values = np.array(values)
    max_value = max(values)
    min_value = min(values)
    values = (values - min_value) / (max_value - min_value)

    policy = np.zeros(2)
    for i in range(num_permutations):
        idx = np.argmax(action_histories[0][i])
        policy[idx] += values[i, 0]
    policy = softmax(policy / temperature)

    return policy


def play_game(env, model: Model, action_space_size: int, num_unroll: int, discount: float) -> Game:
    game = Game(env, discount=discount)
    temperature = 1

    while not game.terminal:
        observation = game.make_image(-1)
        cc = np.random.random()
        if cc < 0.05:
            policy = np.ones(action_space_size) / action_space_size
        else:
            policy = naive_search(model, observation,
                                  action_space_size, num_unroll, temperature)
        game.policy_step(policy, stochastic=False)
    return game

# def play_game(env, model: Model, discount: float):
#     game = Game(env, discount=discount)
#     # num_simulations = 10
#     # dirichlet_alpha = 0.03
#     # exploration_fraction = 0.25
#     temperature = 1

#     while not game.terminal:
#         # root = Node(0)
#         # observation = game.make_image(-1)
#         # expand_node(root, model.initial_inference(observation))
#         # add_exploration_noise(root, dirichlet_alpha, exploration_fraction)
#         # run_mcts(root, game.action_history, model, num_simulations)
#         # action, policy = select_action(root, temperature)
#         # game.apply(action, policy)
