from collections import defaultdict

import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


MAP_20x20 = [
    'SFFFFFFFHFFFFFFHFHFF',
    'FFFFFFFFFFFHFFFFFHFF',
    'FFHFFHFHFFFFFFHFFFFH',
    'FFHFFHFFFFFFFFHFFHFF',
    'FFFHFFFFFFFFFFFFFFHF',
    'FFFFHFFFFFHFFFFHFFFH',
    'FFFFFFFHFHFFHFFFFFFF',
    'HFHFFFFFFFFFFHFFFFFF',
    'HFFFFFFFFHHFHFFHHFFF',
    'FFFFFFFFFHFHFFFFFFFF',
    'FFFFFFFFFFFFHFFFFFFH',
    'FFFFFFFHFFFFFFFFFFFH',
    'FFFFFFHFFFFFFFFFHHFF',
    'HFFHFFFHHFHFFFHHFFFF',
    'FFFFFFFFFHFHFFHHHFFF',
    'HFFFFFHFFFFFHFHFFFFF',
    'HFFFFFFFFFFFFFFFHFFH',
    'FHFFFFFFFHFFFFFFFFFF',
    'FFHFFFFFFFHFFFFHFHFF',
    'FFHFHFFFFFFFHHFFFFFG'
]


class RTDP:
    def __init__(self, env):
        self.env = env
        self.iterations = 17000
        self.gamma = 1.
        self.goal_cost = -1.
        self.hole_cost = .1
        self.default_cost = .1

        self.actions = [0, 1, 2, 3]

        # FrozenLakeEnv gives us a transition matrix, the states and actions.
        # But no costs. So let's set something up ourselves.
        self._init_cost()
        self.calc_policy()

    def calc_policy(self):
        # RTDP
        V = defaultdict(lambda: 0.)

        for i in range(self.iterations):
            if i % 500 == 0:
                print(i)
            obs = self.env.reset()
            while True:
                action = np.argmin(
                    [
                        self.cost[obs][a] + self.gamma * sum(
                            item[0] * V[item[1]]
                            for item in self.env.P[obs][a]
                        )
                        for a in self.actions
                    ]
                )
                V[obs] = self.cost[obs][action] + self.gamma * sum(
                    item[0] * V[item[1]]
                    for item in self.env.P[obs][action]
                )
                obs, reward, done, _ = self.env.step(action)
                if done:
                    cost_ = -1
                    if reward == 0.0:
                        cost_ = 1
                    V[obs] = cost_ + self.gamma * sum(
                        item[0] * V[item[1]]
                        for item in self.env.P[obs][action]
                    )
                    break

        print(' ')
        print(V)
        self.V = V

    def _init_cost(self):
        # TODO Code below kinda ugly.
        cost = defaultdict(dict)
        for state in range(self.env.nS):
            for action in self.actions:
                # why 1? its the action itself. Others are slippery outcomes.
                if len(self.env.P[state][action]) == 1:
                    if self.env.P[state][action][0][2] == 1.0:
                        cost[state][action] = self.goal_cost
                    else:
                        cost[state][action] = self.hole_cost
                    continue

                done = self.env.P[state][action][1][3]
                if done and self.env.P[state][action][1][2] == 0.0:
                    cost[state][action] = self.hole_cost
                else:
                    cost[state][action] = self.default_cost

                if done and self.env.P[state][action][1][2] == 1.0:
                    cost[state][action] = self.goal_cost

        self.cost = cost

    def policy(self, state):
        return np.argmin(
            [
                self.cost[state][a] + self.gamma * sum(
                    item[0] * self.V[item[1]]
                    for item in self.env.P[state][a]
                )
                for a in self.actions
            ]
        )


def evaluate(rtdp, env):
    rewards = 0.
    state = env.reset()
    done = False
    steps = 0
    while not done:
        state, reward, done, _ = env.step(rtdp.policy(state))
        rewards += reward
        steps += 1
        if steps > 1e4:
            break

    return rewards


def run():
    env = FrozenLakeEnv(desc=MAP_20x20)
    rtdp = RTDP(env)
    tot_rewards = 0.
    eval_iter = int(1e4)
    for i in range(eval_iter):
        if i % 100 == 0:
            print(i)
        tot_rewards += evaluate(rtdp, env)
    print(tot_rewards / eval_iter)


if __name__ == '__main__':
    run()
