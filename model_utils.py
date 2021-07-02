import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from modules import Conv2d

class HeadBase(nn.Module):
  def __init__(self, args, action_space, conv_output_size):
    super().__init__()
    self.args = args
    self.atoms = args.atoms
    self.action_space = action_space
    self.conv_output_size = conv_output_size


class ResNetBlock(nn.Module):
	def __init__(self, channel, kernel_size):
		super().__init__()
		self.conv1 = Conv2d(channel, channel, kernel_size)
		self.conv2 = Conv2d(channel, channel, kernel_size)

	def forward(self, x):
		y = F.relu(self.conv1(x))
		y = self.conv2(y)
		return F.relu(y + x)


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.5):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()
    self._stochastic = False

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size, device=self.weight_mu.device)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self._stochastic:
      return self._forward_stochastic(input)
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)

  def _forward_stochastic(self, input):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    weight_epsilon = epsilon_out.ger(epsilon_in)
    bias_epsilon = epsilon_out
    return F.linear(input, self.weight_mu + self.weight_sigma * weight_epsilon, self.bias_mu + self.bias_sigma * bias_epsilon)

  def stochastic(self):
    self._stochastic = True

  def deterministic(self):
    self._stochastic = False