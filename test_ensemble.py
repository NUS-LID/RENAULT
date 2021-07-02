# -*- coding: utf-8 -*-
from __future__ import division
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange

from env import Env

import ensemble


def act_e_greedy(self, state, epsilon=0.001, tid=None):  # High ε can reduce evaluation scores drastically
  if 'mean' in tid:
    greedy_action = ensemble.mean_q(self.online_net, state).argmax(1).item()
  else:
    model = self.online_net.models[int(tid)]
    q = (model(state.unsqueeze(0)) * self.support).sum(2)
    greedy_action = q.argmax(1).item()
  return np.random.randint(0, self.action_space) if np.random.random() < epsilon else greedy_action


# Test DQN
def test(args, dqn, tid):
  dqn.eval()  # Set DQN (online network) to evaluation mode

  env = Env(args)
  env.eval()
  T_rewards = []

  # Test performance over several episodes
  done = True
  t = trange(args.evaluation_episodes)
  for _ in t:
    while True:
      if done:
        state, reward_sum, done = env.reset(), 0, False

      action = act_e_greedy(dqn, state, tid=tid)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()

      if done:
        T_rewards.append(reward_sum)
        t.set_description("Episode reward: {}".format(reward_sum))
        t.refresh() # to show immediately the update
        break
  env.close()

  avg_reward = sum(T_rewards) / len(T_rewards)

  # Return average reward and Q-value
  return avg_reward


def test_ensemble(args, dqn, results_dir):
  tids = [str(i) for i in range(args.n_member)] + ['mean']

  run_id = os.environ.get('RUN_ID', '')
  env_tids = os.environ.get('TIDS', None)
  if env_tids is not None:
    tids = env_tids.split(',')

  for tid in tids:
    tid_label = "tid-{}".format(tid)
    print('tid:', tid_label)
    avg_reward = test(args, dqn, tid)  # Test

    with open(os.path.join(results_dir, str(run_id)+'ensemble_rewards.tsv'), "a") as f:
      f.write("{}\t{}\t{}\n".format(args.game, tid_label, avg_reward))