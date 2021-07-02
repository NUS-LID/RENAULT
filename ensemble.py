import torch
import torch.nn.functional as F


def qs_function(self, state): # -> models x 1 x actions
  with torch.no_grad():
    qs = []
    for model in self.models:
      q = (model(state.unsqueeze(0)) * self.support).sum(2)
      qs.append(q)
    return torch.stack(qs)

def distrib_qs_function(self, x, log=False): # -> models x 1 x actions x atoms
  qs = []
  for model in self.models:
    q = model(x, log=log)
    qs.append(q)
  return torch.stack(qs)


# ---

def mean_q(self, state):
  self.deterministic()
  return qs_function(self, state).mean(0)

# ---


def distrib_mean_q(self, state, log=False):
  self.deterministic()
  return distrib_qs_function(self, state, log=log).mean(0)


# ---


def forward(self, x, log=False):
  return distrib_mean_q(self, x, log=log)


def q_function(self, state):
  return mean_q(self, state)


def act(self, state):
  if self.args.policy == 'mean':
    return self.q(state).argmax(1).item()
  else:
    raise NotImplementedError