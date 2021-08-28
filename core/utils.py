from __future__ import annotations
from abc import abstractmethod, ABC
import math
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, NamedTuple, TYPE_CHECKING
if TYPE_CHECKING:
    # avoid circular imports to get types
    from .model import MuZeroConfig, AbstractNetwork, NetworkOutput, SharedStorage

_pb_c_base = 19652
_pb_c_init = 1.25


class Action:
    def __init__(self, index: int) -> None:
        self.index = index

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other) -> bool:
        return self.index == other.index

    def __gt__(self, other) -> bool:
        return self.index > other.index


class ActionHistory:
    def __init__(self, history: List[Action], action_space_size: int) -> None:
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action) -> None:
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]


class Node:
    def __init__(self, prior: float) -> None:
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children: Dict[Action, Node] = {}
        self.encoded_state = None
        self.reward = 0

    @property
    def expanded(self) -> bool:
        # is not leaf --> inverse of is_leaf
        return len(self.children) > 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


KnownBounds = NamedTuple('KnownBounds', [('min', float), ('max', float)])


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


class AbstractGame(ABC):
    """
    Abstract class that allows to implement a game.
    One instance represent a single episode of interaction with the environment.
    """

    def __init__(self, discount: float):
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def apply(self, action: Action):
        reward = self.step(action)
        self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(i) for i in range(self.action_space_size))
        self.child_visits.append(
            [root.children[a].visit_count / sum_visits if a in root.children else 0 for a in action_space])
        self.root_values.append(root.value)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int):
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * \
                    self.discount ** td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += reward * self.discount ** i
            if current_index < len(self.root_values):
                targets.append(
                    (value, self.rewards[current_index], self.child_visits[current_index]))
            else:
                targets.append((0, 0, []))
        return targets

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    # Methods to be implemented by the children class
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Return the size of the action space."""
        pass

    @abstractmethod
    def step(self, action: Action) -> int:
        """Execute one step of the game conditioned by the given action."""
        pass

    @abstractmethod
    def terminal(self) -> bool:
        """Is the game is finished?"""
        pass

    @abstractmethod
    def legal_actions(self) -> List[Action]:
        """Return the legal actions available at this instant."""
        pass

    @abstractmethod
    def make_image(self, state_index: int):
        """Compute the state of the game."""
        pass


class ReplayBuffer:

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game: AbstractGame):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]

        return [(g.make_image(i), g.history[i:i + num_unroll_steps],
                 g.make_target(i, num_unroll_steps, td_steps))
                for (g, i) in game_pos]

    def sample_game(self) -> AbstractGame:
        # Sample game from buffer either uniformly or according to some priority.
        return np.random.choice(self.buffer)

    def sample_position(self, game: AbstractGame) -> int:
        # Sample position from game either uniformly or according to some priority.
        return np.random.choice(len(game.history))


def ucb_score(parent: Node, child: Node, min_max_stats: MinMaxStats, pb_c_base: Optional[float] = None, pb_c_init: Optional[float] = None):
    pb_c_base = _pb_c_base if pb_c_base is None else pb_c_base
    pb_c_init = _pb_c_init if pb_c_init is None else pb_c_init

    pb_c = math.log((parent.visit_count + pb_c_base + 1) /
                    pb_c_base) + pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value)
    return prior_score + value_score


def select_child(node: Node, min_max_stats: MinMaxStats):
    # print([(ucb_score(node, child, min_max_stats), action, child)
    #       for action, child in node.children.items()])
    # can specify ucb params here
    _, action, child = max((ucb_score(node, child, min_max_stats), action, child)
                           for action, child in node.children.items())
    return action, child


def expand_node(node: Node, actions: List[Action], network_output: NetworkOutput):
    # add all possible edges to node
    node.encoded_state = network_output.encoded_state
    node.reward = network_output.reward

    # assign softmax to each node
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


def backpropagate(search_path: List[Node], value: float, discount: float, min_max_stats: MinMaxStats):
    for node in search_path:
        node.value_sum += value
        node.visit_count += 1
        min_max_stats.update(node.value)

        value = node.reward + discount * value


def run_mcts(config: MuZeroConfig, root: Node, action_history: ActionHistory, network: AbstractNetwork):
    min_max_stats = MinMaxStats(config.known_bounds)

    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        # traverse tree till a leaf node is hit
        while node.expanded:
            # print(_, 'hi')
            action, node = select_child(node, min_max_stats)
            # print(action.index)
            history.add_action(action)
            search_path.append(node)

        # print(history.history)

        # get leaf node's parent and use g and f to get r, s, p, v
        parent = search_path[-2]
        network_output = network.recurrent_inference(
            parent.encoded_state, action)
        # print(network_output)
        # reward, hidden_state, policy, value = network.recurrent_inference()

        expand_node(node, history.action_space(), network_output)

        backpropagate(search_path, network_output.value,
                      config.discount, min_max_stats)


def softmax_sample(visits: List[Tuple[int, Action]], t: float):
    visit_counts, actions = zip(*visits)
    visit_counts_exp = np.exp(visit_counts) * (1 / t)
    probs = visit_counts_exp / visit_counts_exp.sum(axis=0)

    # this looks weird
    action_idx = np.random.choice(len(actions), p=probs)
    return actions[action_idx]


def select_action(config: MuZeroConfig, n_moves: int, node: Node, network: AbstractNetwork):
    visits = [(child.visit_count, action)
              for action, child in node.children.items()]
    t = config.visit_softmax_temperature_fn(
        num_moves=n_moves, training_steps=network.training_steps())
    action = softmax_sample(visits, t)
    return action


def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n, in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


def play_game(config: MuZeroConfig, network: AbstractNetwork) -> AbstractGame:
    game = config.new_game()
    count = 0

    while not game.terminal and len(game.history) < config.max_moves:
        count += 1
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.legal_actions(),
                    network.initial_inference(current_observation))

        # print(root.children)
        nodes = root.children.values()
        # print([i.prior for i in nodes])
        add_exploration_noise(config, root)
        # print([i.prior for i in nodes])

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        # print(game.action_history().history)
        run_mcts(config, root, game.action_history(), network)
        # print([i.prior for i in root.children.values()])
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
        # print(action.index)
        # if count == 2:
        #     quit()
        # print(game.terminal, game.make_image(-1))
    return game


def selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    network = storage.latest_network()
    episode_returns = []
    for episode in range(config.num_episodes):
        game = play_game(config, network)
        replay_buffer.save_game(game)
        episode_returns.append(sum(game.rewards))

    return sum(episode_returns) / (episode + 1)


def update_weights(config, batch, network, optimizer):
    network.train()
    p_loss, v_loss = 0, 0
    for observation, actions, targets in batch:
        network_output = network.initial_inference(observation, True)
        predictions = [
            (network_output.value, network_output.reward, network_output.policy_logits)]
        # print(network_output.encoded_state[0])

        for action in actions:
            network_output = network.recurrent_inference(
                network_output.encoded_state, action, True)
            predictions.append(
                (network_output.value, network_output.reward, network_output.policy_logits))

        for prediction, target in zip(predictions, targets):
            if (len(target[2]) > 0):

                value, reward, policy_logits = prediction
                target_value, target_reward, target_policy = target
                # print(policy_logits)

                target_policy = torch.Tensor(
                    target_policy, device=config.device)
                target_value = torch.Tensor(
                    [target_value], device=config.device)

                # print(policy_logits, torch.log(policy_logits))
                # print(policy_logits)
                p_loss += torch.sum(-target_policy *
                                    torch.log(policy_logits))
                v_loss += torch.sum((target_value - value) ** 2)

        optimizer.zero_grad()
        total_loss = p_loss + v_loss
        total_loss.backward()
        optimizer.step()
        network.steps += 1

        # print(p_loss, p_loss.item())

        return p_loss.item(), v_loss.item()


def train_network(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, optim):

    last_network = storage.latest_network()
    network = storage.new_network()
    network.load_state_dict(last_network.state_dict())

    optimizer = optim(network.parameters(), lr=0.001)
    # copy over latest weights

    for epoch in range(config.num_epochs):
        batch = replay_buffer.sample_batch(
            config.num_unroll_steps, config.td_steps)
        p_loss, v_loss = update_weights(config, batch, network, optimizer)
        # print(' %10.2f %10.2f' % (p_loss, v_loss))
    return network
