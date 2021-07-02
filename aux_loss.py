import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from distributional import create_p_categorical
from model_utils import HeadBase, ResNetBlock, NoisyLinear
from modules import Conv2d

from functools import partial

# ---

def dist(a, b):
  batch = a.shape[0]
  assert batch == b.shape[0]
  return F.pairwise_distance(a.view(batch, -1), b.view(batch, -1))

# ---

class LossBase(HeadBase):
  def __init__(self, name, tid, args, action_space, conv_output_size):
    super().__init__(args, action_space, conv_output_size)
    self.class_name = self.__class__.__name__
    self.name = name
    self.tid = tid


class InverseDynamicLoss(LossBase):
  def __init__(self, name, tid, args, action_space, conv_output_size):
    super().__init__(name, tid, args, action_space, conv_output_size)
    self.fc_h = nn.Linear(self.conv_output_size * 2, args.hidden_size)
    self.fc_z = nn.Linear(args.hidden_size, action_space)

  def forward(self, feat1, feat2, actions):
    x1, x2 = feat1.view(-1, self.conv_output_size), feat2.view(-1, self.conv_output_size)
    x = torch.cat([x1, x2], dim=1)
    a = self.fc_z(F.relu(self.fc_h(x)))
    return F.cross_entropy(a, actions, reduction='none')


class CategoricalRewardLoss(LossBase):
  def __init__(self, name, tid, args, action_space, conv_output_size):
    super().__init__(name, tid, args, action_space, conv_output_size)
    self.fc_z = nn.Linear(args.hidden_size * 2, action_space * self.args.reward_atoms)
    self.distrib = create_p_categorical(a=-1, b=1, n=args.reward_atoms, sigma=args.reward_sigma)

  def forward(self, hv, ha, actions, rewards):
    with torch.no_grad():
      rewards = self.distrib(rewards.squeeze())
    r = self.fc_z(torch.cat([hv,ha], dim=1)).view(-1, self.action_space, self.args.reward_atoms)
    r = F.log_softmax(r, dim=2)
    r = r[range(self.args.batch_size), actions]
    return -torch.sum(rewards * r, 1)


class CategoricalIntensityLoss(LossBase):
  def __init__(self, name, tid, args, action_space, conv_output_size):
    super().__init__(name, tid, args, action_space, conv_output_size)
    linear = '-l' in name
    if linear:
      print('Linear CI')
      self.net = nn.Linear(self.conv_output_size, action_space * self.args.intensity_atoms)
    else:
      self.net = nn.Sequential(
                      nn.Linear(self.conv_output_size, args.hidden_size), nn.ReLU(),
                      nn.Linear(args.hidden_size, action_space * self.args.intensity_atoms)
                    )

    self.distrib = create_p_categorical(a=0, b=84, n=args.intensity_atoms, sigma=args.intensity_sigma)

  def intensity(self, x1, x2):
    assert x1.min() >= 0 and x1.max() <= 1
    diff = dist(x1.mean(1), x2.mean(1)).squeeze()
    assert torch.all(diff <= 84)
    return diff

  def forward(self, x, actions, x1, x2):
    with torch.no_grad():
      intensities = self.intensity(x1, x2)
      intensities = self.distrib(intensities.squeeze())
    i = self.net(x).view(-1, self.action_space, self.args.intensity_atoms)
    i = F.log_softmax(i, dim=2)
    i = i[range(self.args.batch_size), actions]
    return -torch.sum(intensities * i, 1)


class LatentNextStateLoss(LossBase):
  def __init__(self, name, tid, args, action_space, conv_output_size):
    super().__init__(name, tid, args, action_space, conv_output_size)
    self.conv1 = Conv2d(65,64,3)
    self.rn1 = ResNetBlock(64,3)

  def forward(self, feat1, feat2, actions):
    b,c,h,w = feat1.shape
    action_channel = torch.ones((b,1,h,w), device=feat1.device) \
                      * actions.unsqueeze(1).unsqueeze(1).unsqueeze(1) / (self.action_space-1)
    x = torch.cat([feat1, action_channel], dim=1)
    x = F.relu(self.conv1(x) + feat1)
    x = self.rn1(x)
    return F.smooth_l1_loss(x, feat2, reduction='none')


class DiscountModel(HeadBase):
  def __init__(self, args, action_space, conv_output_size):
    super().__init__(args, action_space, conv_output_size)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def q(self, hv, ha, log=False):
    v = self.fc_z_v(hv).view(-1, 1, self.atoms)
    a = self.fc_z_a(ha).view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)
    if log:
      q = F.log_softmax(q, dim=2)
    else:
      q = F.softmax(q, dim=2)
    return q

  def forward(self, x, log=False, model=None):
    out = model(x, log=log, return_tuple=True)
    return self.q(out['hv'], out['ha'], log=log)


class DiscountLoss(LossBase):
  def __init__(self, name, tid, args, action_space, conv_output_size):
    super().__init__(name, tid, args, action_space, conv_output_size)
    self.discount = float(name.split('-')[-1])
    self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.args.multi_step)], dtype=torch.float32, device=self.args.device)
    self.net = DiscountModel(args, action_space, conv_output_size)
    if args.discount_v_scaling:
      max_reward = 0.1 # weird setup from the original
      def v_max(max_reward, discount):
        return round(max_reward/(1-discount), 3)
      self.Vmin = -v_max(max_reward, self.discount)
      self.Vmax = v_max(max_reward, self.discount)
      self.support = torch.linspace(self.Vmin, self.Vmax, args.atoms).to(device=args.device)
      self.delta_z = (self.Vmax - self.Vmin) / (args.atoms - 1)

  def forward(self, hv, ha, actions, rewards, next_states, nonterminals, weights, agent):
    returns = torch.matmul(rewards, self.n_step_scaling)

    log_ps = self.net.q(hv, ha, log=True)
    log_ps_a = log_ps[range(self.args.batch_size), actions]

    _online_net = agent.online_net.get_aux(self.tid, self.name).net
    _target_net = agent.target_net.get_aux(self.tid, self.name).net
    online_net = partial(_online_net, model=agent.online_net.models[self.tid])
    target_net = partial(_target_net, model=agent.target_net.models[self.tid])

    if self.args.discount_v_scaling:
      m = agent.compute_target(online_net, target_net, actions, returns, next_states, nonterminals, 
                              discount=self.discount, Vmin=self.Vmin, Vmax=self.Vmax, support=self.support, delta_z=self.delta_z)
    else:
      m = agent.default_compute_target(online_net, target_net, actions, returns, next_states, nonterminals, discount=self.discount)

    q_loss = -torch.sum(m * log_ps_a, 1)
    q_loss = (weights * q_loss)
    return q_loss


class MomentChangesLoss(LossBase):
  def __init__(self, name, tid, args, action_space, conv_output_size):
    super().__init__(name, tid, args, action_space, conv_output_size)
    self.net = nn.Sequential(
                    nn.Linear(self.conv_output_size, args.hidden_size), nn.ReLU(),
                    nn.Linear(args.hidden_size, action_space)
                  )
    x = torch.tensor([i for i in range(84)]) + 1
    y = torch.tensor([i for i in range(84)]) + 1
    grid = torch.stack(torch.meshgrid(x, y)).to(args.device).float()
    self.distance = torch.norm(grid - torch.zeros_like(grid), dim=0)
    self.distance = self.distance / self.distance.max() # normalize distance
    self.moment_max = self.distance.sum().sqrt() # take sqrt, otherwise value is too small

  def moment(self, q, normalize=True):
    moment = (self.distance * q).sum(-1).sum(-1).mean(-1) # mean over channel
    if normalize:
      return moment / self.moment_max
    return moment

  def moment_changes(self, x1, x2):
    assert x1.min() >= 0 and x1.max() <= 1
    diff = self.moment(x2) - self.moment(x1)
    return diff

  def forward(self, x, actions, x1, x2):
    with torch.no_grad():
      moment_changes = self.moment_changes(x1, x2)
    m = self.net(x).view(-1, self.action_space)
    m = m[range(self.args.batch_size), actions]
    return F.smooth_l1_loss(m, moment_changes, reduction='none')


def get_loss_by_name(name):
  if name == 'inverse_dynamic' or name == 'id':
    return InverseDynamicLoss
  elif name == 'categorical_reward' or name == 'cr':
    return CategoricalRewardLoss
  elif name == 'moment_changes' or name == 'mc':
    return MomentChangesLoss
  elif 'categorical_intensity' in name or 'ci' in name:
    return CategoricalIntensityLoss
  elif name == 'latent_next_state' or name == 'lns':
    return LatentNextStateLoss
  elif 'discount' in name or 'dsc' in name:
    return DiscountLoss
  else:
    raise NotImplementedError

def get_aux_loss(name, *args):
  return get_loss_by_name(name)(name, *args)