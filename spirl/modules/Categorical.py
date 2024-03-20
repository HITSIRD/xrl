import torch
import math
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal

from spirl.utils.pytorch_utils import ten2ar
from spirl.utils.general_utils import batch_apply


class Categorical:
    """ Represents a categorical distribution """

    # TODO: implement a dict conversion function
    def __init__(self, probs=None, logits=None, codebook=None):
        self.prob = torch.distributions.Categorical(probs=probs, logits=logits)
        self.codebook = codebook

        # if codebook is not None:
        #     self.codebook.embedding.weight.requires_grad = False

    def sample(self):
        if self.codebook is not None:
            index = self.prob.sample()
            return self.codebook.embedding.weight[index], index, self.prob.log_prob(index)
        else:
            return self.prob.sample()

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p)"""
        delta = 1e-15
        return torch.nn.functional.kl_div((other.prob.probs + delta).log(), self.prob.probs + delta, reduction='none')

    def nll(self, x):
        # Negative log likelihood (probability)
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        if isinstance(val, tuple):
            val = val[1]
        return self.prob.log_prob(val)

    def entropy(self):
        return self.prob.entropy()

    @property
    def shape(self):
        return self.prob.probs.shape

    def rsample(self):
        """Identical to self.sample(), to conform with pytorch naming scheme."""
        return self.sample()

    def detach(self):
        """Detaches internal variables. Returns detached Gaussian."""
        return type(self)(logits=self.prob.logits.detach(), codebook=self.codebook)

    def to_numpy(self):
        """Convert internal variables to numpy arrays."""
        return self