"""PyTorch OPs that stochastically gate (on/off) activations using sampling.
A set of Tensorflow OPs that provide stochastic gating (on/off) functionality.
This is done using the Gumbel-Softmax and Logistic-Sigmoid reparameterization
tricks. The gating probability is trainable.
References:
[1] Categorical Reparameterization with Gumbel-Softmax:
    https://arxiv.org/abs/1611.01144
[2] Fine-Grained Stochastic Architecture Search:
    https://arxiv.org/abs/2006.09581
"""

import torch
import torch.nn as nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.uniform import Uniform
from torch.distributions.transforms import SigmoidTransform

class LogisticSigmoidGating(nn.Module):

    def __init__(self,
                 axis=1,
                 temperature=0.001,
                 straight_through=False,
                 log_odds_init=2.5,
                 soft_mask_for_inference=True,
                 keep_first_channel_alive=True,
                 annealing_rate=None,
                 global_step=None,
                 name=None):
        """
        :param axis: The axis on which to gate. E.g., for [b,C,w,h] tensor and axis=1,
        the mask tensor will be of shape [1,C,1,1].
        :param temperature: Float scaler. A variable that controls sampling. See [1] and [2] for details.
        :param straight_through: Bool. If True, using Straight Through sampling, in which the forward pass is discrete, a
        and the backward pass uses the gradients of a differentiable approximation to the gate. This arg should not change
        when calling logistic_sigmoid_gating multiple times for the same graph, because there can be only one LogisticSigmoidGating op.
        :param log_odds_init: Value to initialize the log odds ratio (logits).
        logits = log(p/1-p). Default: 2.5 => 92% of being 'on'
        :param soft_mask_for_inference: multiply the mask by logits for inference.
        :param keep_first_channel_alive: If True, the first channel is always alive
        (unmasked). If False, it is possible to sample a mask of all zeros.
        :param annealing_rate: If provided, the temperature is exponentially decayed by a factor of 0.1 every 'annealing_rate'.
        The larger the smoother. The smaller, the more "one-hot".
        :param global_step: Required if decaying temperature.
        :param name: Layer name.
        """
        super(LogisticSigmoidGating, self).__init__(name=name)

        self.axis = axis
        self.temperature = temperature
        self.straight_through = straight_through
        self.log_odds_init = log_odds_init
        self.soft_mask_for_inference = soft_mask_for_inference
        self.keep_first_channel_alive = keep_first_channel_alive
        self.annealing_rate = annealing_rate
        self.global_step = global_step

        assert isinstance(self.log_odds_init, (int, float)), 'log_odds_init has unsupported value.'

        if self.annealing_rate:
            if self.global_step is None:
                raise ValueError('Must provide global step if decaying temperature.')
            self.temperature = self.temperature * (
                0.1 ** (self.global_step / float(self.annealing_rate)))

    def forward(self, activation, is_training):
        """Build and apply stochastic mask on activation"""
        """
        activation: 4D tensor
        is_training: if False, no sampling is done. Gating is deterministically performed if the learned log_odds > 0 (p>0.5).
        """
        mask_len = activation.shape[self.axis]
        logits = nn.Parameter(torch.rand(activation.shape))
        nn.init.constant_(logits, self.log_odds_init)

        if is_training:
            mask = _logistic_sigmoid_sample(
                logits, self.temperature, self.straight_through
            )
        else:
            mask = logits > 0.0
            if self.soft_mask_for_inference:
                # Like dropout we multiply the mask by the prob
                mask *= torch.sigmoid(logits)

        # if self.keep_first_channel_alive:

        return mask * activation

def logistic_sigmoid_gating(activation,
                            axis,
                            is_training,
                            temperature=0.001,
                            straight_through=False,
                            log_odds_init=2.5,
                            soft_mask_for_inference=True,
                            keep_first_channel_alive=True,
                            annealing_rate=None,
                            global_step=None,
                            scope=None):
  """Apply logistic-sigmoid gating (wrapper for LogisticSigmoidGating Layer)."""
  layer_fn = LogisticSigmoidGating(
      axis=axis,
      temperature=temperature,
      straight_through=straight_through,
      log_odds_init=log_odds_init,
      soft_mask_for_inference=soft_mask_for_inference,
      keep_first_channel_alive=keep_first_channel_alive,
      annealing_rate=annealing_rate,
      global_step=global_step,
      name=scope)
  return layer_fn(activation, is_training=is_training)

def Logistic(a,b):
    return TransformedDistribution(Uniform(0, 1), [SigmoidTransform().inv, AffineTransform(a,b)])

def _logistic_sigmoid_sample(logits, tau, straight_through):
    """
    :param logits: The log odds ratio of the Bernoulli probability. i.e., log(p/1-p).
    :param tau: The temperature variable that controls sampling.
    :param straight_through: If True, the forward pass is discrete, and the backward pass uses the gradients of
    a differentiable approximation to the gate.
    :return: The sampled mask (gating) tensor
    """
    logistic_dist = Logistic(0.0, 1.0)
    logistic_sample = logistic_dist.sample(logits.shape) # the noise

    if straight_through:
        mask = (logits + logistic_sample) > 0.0
    else:
        mask = nn.Sigmoid((logits + logistic_sample) / tau)
    return mask

