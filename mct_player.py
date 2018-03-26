import config as cfg
import numpy as np

class Node:
    def __init__(self, parent = None, Priori_probability = 0.0):
        # prio_p means Priori_probability, this can be get from Neural Network
        # value means average action value accumulation from this node, this is calculated by Monte Carlo Search
        # visit means the number this node has been visited, used to calculate value
        # c_puct Exploration parameter
        self.prio_p = priori_probability
        self.value = 0.0
        self.visit = 0
        self.c_puct = cfg.c_puct

        self.parent = parent
        self.children = {}

    def expand(self, node, action):
        if action in self.children:
            pass
        self.children[action] = node

    def get_node_value(self):
        # The total value is a trade-off between Exploration & Exploitation
        # u means the Exploration value, it is related with a Priori_probability
        # and negative related with the number this node has been visited
        # Neighborhood node also contribute to this value
        # self.value means the Exploitation value
        u = self.c_puct * self.prio_p * (np.sqrt(self._parent.visit) / (1 + self.visit))
        return self.value + u

    def ε_select(self,ε = cfg.ε):
        assert(ε >= 0)
        assert(ε <= 1)
        num = len(self.children)
        assert(num > 0)
        values = list(self.children.values())
        max_value = max(values)

        p = np.zeros([num])
        for i in range(num):
            if values[i] == max_value:
                p[i] = (1 - ε + ε / num)
            else:
                p[i] = (ε / num)
        p = p / sum(p)
        action = np.random.choice(self.children.keys(),1,p = p)[0]
        return action, self.children[action]

    def greedy_select(self):
        return self.ε_select(ε = 0)


class MC_Tree:
    def __init__(self, policy_fn, evaluation_fn):
        self.root = Node()
        self.evaluation_fn = evaluation_fn

    def search(self, board):
        # Monte Carlo Tree Search
        current_node = self.root
        while True:
            if len(current_node.children) > 0:
                # This is not leaf node
                # we can select one child node and go through it
                action,current_node = current_node.ε_select()
                board.move(action)
            else:
                # This is leaf node
                # prepare to expand the tree
                break
        # use your evaluation function(Neural network) to estimate a Priori_probability at this state
        # and the value for this leaf state
        action_probs, state_value = self.evaluation_fn(board)
        game_end,last_player_win = board.end_game()
        if game_end:
            if last_player_win:




class MCT_player:
    def __init__(self,policy_fn,evaluation_fn):
        self.policy_fn = policy_fn
        self.evaluation_fn = evaluation_fn
        self.mc_tree = MC_Tree()


def ε_select(child):
    ε = cfg.ε
    num = len(child)
    assert (num > 0)
    values = list(child.values())
    max_value = max(values)

    p = np.zeros([num])
    for i in range(num):
        if values[i] == max_value:
            p[i] = (1 - ε + ε / num)
        else:
            p[i] = (ε / num)
    p = p / sum(p)
    action = np.random.choice(list(child.keys()), 1, p=p)[0]
    return action, child[action]