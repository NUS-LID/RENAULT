# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import MultiDQN


class Agent():
  def __init__(self, args, env):
    self.args = args
    self.action_space = env.action_space()
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    self.online_net = MultiDQN(args, self.action_space, self.support).to(device=args.device)
    if args.model:  # Load pretrained model if provided
      if os.path.isfile(args.model):
        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
        if 'conv1.weight' in state_dict.keys():
          for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'), ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'), ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
            state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
            del state_dict[old_key]  # Delete old keys for strict load_state_dict
        self.online_net.load_state_dict(state_dict)
        print("Loading pretrained model: " + args.model)
      else:  # Raise error if incorrect model path provided
        raise FileNotFoundError(args.model)

    self.online_net.train()

    self.target_net = MultiDQN(args, self.action_space, self.support).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch)
  def act(self, state):
    with torch.no_grad():
      return self.online_net.act(state)
      # return self.online_net.q(state).argmax(1).item()
      # return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

  # Acts with an ε-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
    return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

  def default_compute_target(self, online_net, target_net, actions, returns, next_states, nonterminals, discount=None):
    discount = discount or self.discount
    return self.compute_target(online_net, target_net, actions, returns, next_states, nonterminals, 
                              discount=discount, Vmin=self.Vmin, Vmax=self.Vmax, support=self.support, delta_z=self.delta_z)

  def compute_target(self, online_net, target_net, actions, returns, next_states, nonterminals, discount=None, Vmin=None, Vmax=None, support=None, delta_z=None):
    assert discount is not None and Vmin is not None and Vmax is not None and support is not None and delta_z is not None
    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      
      pns = target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (discount ** self.n) * support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=Vmin, max=Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - Vmin) / delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = next_states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

      return m

  def learn(self, mem):
    loss = 0
    losses = {}

    for tid in range(self.args.n_member):
      online_net = self.online_net.models[tid]
      online_net.deterministic() # individual model must be deterministic
      if self.args.shared_target:
        target_net = self.target_net # stochasticity is determined by ensemble config
      else:
        target_net = self.target_net.models[tid]
        target_net.deterministic() # individual model must be deterministic

      # Sample transitions
      idxs, states, actions, returns, next_states, nonterminals, weights, d_rewards, d_next_states, rewards = mem.sample(tid, self.batch_size)

      # Calculate current state probabilities (online network noise already sampled)
      online_out = online_net(states, log=True, return_tuple=True)
      log_ps = online_out['q']  # Log probabilities log p(s_t, ·; θonline)
      log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

      target_net.reset_noise()  # Sample new target net noise

      m = self.default_compute_target(online_net, target_net, actions, returns, next_states, nonterminals)

      q_loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

      priority_weight = q_loss.detach().cpu().numpy()
      mem.update_priorities(tid, idxs, priority_weight)  # Update priorities of sampled transitions

      losses[f'[{tid}-q]'] = q_loss.detach().mean().cpu().numpy()

      q_loss_mean = (weights * q_loss).mean()

      loss += q_loss_mean

      # ---

      aux_losses = []
      auxs = self.online_net.auxs[tid]
      for aux in auxs:
        if aux.class_name == 'CategoricalRewardLoss':
          aux_loss = aux(online_out['hv'], online_out['ha'], actions, d_rewards)
        elif aux.class_name == 'InverseDynamicLoss':
          feat2 = online_net.conv(d_next_states)
          aux_loss = aux(online_out['feat'], feat2, actions)
        elif aux.class_name == 'LatentNextStateLoss':
          feat2 = online_net.conv(d_next_states)
          aux_loss = aux(online_out['feat'], feat2, actions)
        elif aux.class_name == 'CategoricalIntensityLoss':
          aux_loss = aux(online_out['x'], actions, states, d_next_states)
        elif aux.class_name == 'MomentChangesLoss':
          aux_loss = aux(online_out['x'], actions, states, d_next_states)
        elif aux.class_name == 'DiscountLoss':
          aux_loss = aux(online_out['hv'], online_out['ha'], actions, rewards, next_states, nonterminals, weights, self)
        else:
          raise NotImplementedError
        if not self.args.hide_aux_loss:
          losses[f'{tid}-{aux.name}'] = aux_loss.detach().mean().cpu().numpy()
        # loss += aux_loss.mean()
        aux_losses.append(aux_loss.mean())

      # handle none
      if len(aux_losses) == 0:
        aux_losses = [torch.tensor(0)]

      aux_losses = torch.stack(aux_losses)
      if self.args.aux_aggregate == 'mean':
        aux_loss_agg = aux_losses.mean()
      elif self.args.aux_aggregate == 'sum':
        aux_loss_agg = aux_losses.sum()
      else:
        raise NotImplementedError

      loss = loss + aux_loss_agg

      # ---

    losses['loss'] = loss.detach().mean().cpu().numpy()

    self.online_net.zero_grad()
    loss.backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    return losses

  def learn_REN(self, mem):
    loss = 0
    losses = {}

    for tid in range(self.args.n_member):
      online_net = self.online_net.models[tid]
      online_net.deterministic() # individual model must be deterministic
      if self.args.shared_target:
        target_net = self.target_net # stochasticity is determined by ensemble config
      else:
        target_net = self.target_net.models[tid]
        target_net.deterministic() # individual model must be deterministic

      # Sample transitions
      idxs, states, actions, returns, next_states, nonterminals, weights, d_rewards, d_next_states, rewards = mem.sample(tid, self.batch_size)

      # Calculate current state probabilities (online network noise already sampled)
      log_ps = online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
      log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

      with torch.no_grad():
        # Calculate nth next state probabilities
        pns = online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
        dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
        argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
        target_net.reset_noise()  # Sample new target net noise
        pns = target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
        pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

        # Compute Tz (Bellman operator T applied to z)
        Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
        # Compute L2 projection of Tz onto fixed support z
        b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        # Distribute probability of Tz
        m = states.new_zeros(self.batch_size, self.atoms)
        offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
        m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
        m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)


      q_loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

      priority_weight = q_loss.detach().cpu().numpy()
      mem.update_priorities(tid, idxs, priority_weight)  # Update priorities of sampled transitions

      losses['q{}_loss'.format(tid)] = q_loss.detach().cpu().numpy().mean()

      q_loss_mean = (weights * q_loss).mean()

      loss += q_loss_mean

    losses['loss'] = loss.detach().cpu().numpy().mean()

    self.online_net.zero_grad()
    loss.backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    return losses

  def learn_REN_J(self, mem):
    loss = 0
    losses = {}

    online_net = self.online_net
    target_net = self.target_net # stochasticity is determined by ensemble config

    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights, d_rewards, d_next_states, rewards = mem.sample(0, self.batch_size)

    # Calculate current state probabilities (online network noise already sampled)
    ps = online_net(states, log=False)  # can't simply take the average over log prob
    ps = ps.clamp(min=1e-12, max=1e12)
    log_ps = torch.log(ps) # log over average
    log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
      argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
      target_net.reset_noise()  # Sample new target net noise
      pns = target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
      pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

      # Compute Tz (Bellman operator T applied to z)
      Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (self.atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = states.new_zeros(self.batch_size, self.atoms)
      offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)


    q_loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))

    priority_weight = q_loss.detach().cpu().numpy()
    mem.update_priorities(0, idxs, priority_weight)  # Update priorities of sampled transitions

    losses['q{}_loss'.format(0)] = q_loss.detach().cpu().numpy().mean()

    q_loss_mean = (weights * q_loss).mean()

    loss += q_loss_mean

    losses['loss'] = loss.detach().cpu().numpy().mean()

    self.online_net.zero_grad()
    loss.backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    return losses

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Save model parameters on current device (don't move model between devices)
  def save(self, path, name='model.pth'):
    torch.save(self.online_net.state_dict(), os.path.join(path, name))

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return self.online_net.q(state).max(1)[0].item()
      # return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
