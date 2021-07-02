# -*- coding: utf-8 -*-
from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F

import ensemble
from aux_loss import get_aux_loss
from model_utils import HeadBase, NoisyLinear

# ---

class Head(HeadBase):
  def __init__(self, args, action_space, conv_output_size):
    super().__init__(args, action_space, conv_output_size)

    self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    self.noisy_modules = [module for name, module in self.named_children() if 'fc' in name]

  def forward(self, x, log=False, return_tuple=False):
    x = x.view(-1, self.conv_output_size)
    hv = F.relu(self.fc_h_v(x))
    ha = F.relu(self.fc_h_a(x))
    v = self.fc_z_v(hv)  # Value stream
    a = self.fc_z_a(ha)  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    if return_tuple:
      return {'x':x, 'v':v, 'a':a, 'hv':hv, 'ha':ha, 'q':q}
    return q

  def reset_noise(self):
    for module in self.noisy_modules:
      module.reset_noise()

  def stochastic(self):
    for module in self.noisy_modules:
      module.stochastic()

  def deterministic(self):
    for module in self.noisy_modules:
      module.deterministic()


class Conv(nn.Module):
  def __init__(self, args):
    super().__init__()
    if args.architecture == 'canonical':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
      self.output_size = 3136
    elif args.architecture == 'data-efficient':
      self.convs = nn.Sequential(nn.Conv2d(args.history_length, 32, 5, stride=5, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 5, stride=5, padding=0), nn.ReLU())
      self.output_size = 576

  def forward(self, x):
    x = self.convs(x)
    return x


class DQN(nn.Module):
  def __init__(self, conv, head):
    super().__init__()
    self.conv = conv
    self.head = head

  def forward(self, x, log=False, return_tuple=False):
    data = {}
    data['raw'] = x
    x = self.conv(x)
    data['feat'] = x
    out = self.head(x, log=log, return_tuple=True) # x,v,a,q
    data = {**data, **out}
    if return_tuple:
      return data
    return data['q']

  def reset_noise(self):
    self.head.reset_noise()

  def stochastic(self):
    self.head.stochastic()

  def deterministic(self):
    self.head.deterministic()


class MultiDQN(nn.Module):
  def __init__(self, args, action_space, support):
    super().__init__()
    self.args = args
    self.support = support
    convs = [Conv(args) for i in range(args.n_member)]
    self.models = nn.ModuleList([DQN(convs[i], Head(args, action_space, convs[i].output_size)) for i in range(args.n_member)])
    
    auxs = []
    for i, aux_names in enumerate(self.args.auxs):
      if aux_names == 'none':
        auxs.append(nn.ModuleList([]))
        continue
      head_aux = nn.ModuleList([get_aux_loss(aux_name, i, args, action_space, convs[i].output_size) for aux_name in aux_names.split('+')])
      auxs.append(head_aux)
    self.auxs = nn.ModuleList(auxs)

  def get_aux(self, tid, name):
    for mod in self.auxs[tid]:
      if mod.name == name:
        return mod

  def forward(self, x, log=False):
    return ensemble.forward(self, x, log=log)

  def q(self, state):
    return ensemble.q_function(self, state)

  def act(self, state):
    return ensemble.act(self, state)

  def reset_noise(self):
    for model in self.models:
      model.reset_noise()
  
  def stochastic(self):
    for model in self.models:
      model.stochastic()

  def deterministic(self):
    for model in self.models:
      model.deterministic()