import numpy as np
import pymc as pm
from pymc.distributions.distribution import Discrete
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.shape_utils import rv_size_is_none
import pytensor.tensor as pt
from pytensor.tensor import TensorConstant
from pytensor.tensor.random.basic import vsearchsorted
from pytensor.tensor.random.op import RandomVariable
import warnings


class CategoricalRV(RandomVariable):
    r"""A categorical discrete random variable.

    The probability mass function of `categorical` in terms of its :math:`N` event
    probabilities :math:`p_1, \dots, p_N` is:

    .. math::

        P(k=i) = p_k

    where :math:`\sum_i p_i = 1`.

    """

    name = "categorical"
    signature = "(p)->()"
    dtype = "int64"
    _print_name = ("Categorical", "\\operatorname{Categorical}")

    def __call__(self, p, size=None, **kwargs):
        r"""Draw samples from a discrete categorical distribution.

        Signature
        ---------

        `(j) -> ()`

        Parameters
        ----------
        p
            An array that contains the :math:`N` event probabilities.
        size
            Sample shape. If the given size is `(m, n, k)`, then `m * n * k`
            independent, identically distributed random samples are
            returned. Default is `None`, in which case a single sample
            is returned.

        """
        return super().__call__(p, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, p, size):
        if size is None:
            size = p.shape[:-1]
        else:
            # Check that `size` does not define a shape that would be broadcasted
            # to `p.shape[:-1]` in the call to `vsearchsorted` below.
            if len(size) < (p.ndim - 1):
                raise ValueError("`size` is incompatible with the shape of `p`")
            for s, ps in zip(reversed(size), reversed(p.shape[:-1])):
                if s == 1 and ps != 1:
                    raise ValueError("`size` is incompatible with the shape of `p`")

        unif_samples = rng.uniform(size=size)
        samples = vsearchsorted(p.cumsum(axis=-1), unif_samples)

        return samples


categorical = CategoricalRV()


class Categorical(Discrete):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        ps = [[0.1, 0.6, 0.3], [0.3, 0.1, 0.1, 0.5]]
        for p in ps:
            x = range(len(p))
            plt.plot(x, p, '-o', label='p = {}'.format(p))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=1)
        plt.show()

    ========  ===================================
    Support   :math:`x \in \{0, 1, \ldots, |p|-1\}`
    ========  ===================================

    Parameters
    ----------
    p : array of floats
        p > 0 and the elements of p must sum to 1.
    logit_p : float
        Alternative log odds for the probability of success.
    """

    rv_op = categorical

    @classmethod
    def dist(cls, p=None, logit_p=None, **kwargs):
        if p is not None and logit_p is not None:
            raise ValueError("Incompatible parametrization. Can't specify both p and logit_p.")
        elif p is None and logit_p is None:
            raise ValueError("Incompatible parametrization. Must specify either p or logit_p.")

        if logit_p is not None:
            p = pm.math.softmax(logit_p, axis=-1)

        p = pt.as_tensor_variable(p)
        if isinstance(p, TensorConstant):
            p_ = np.asarray(p.data)
            if np.any(p_ < 0):
                raise ValueError(f"Negative `p` parameters are not valid, got: {p_}")
            p_sum_ = np.sum([p_], axis=-1)
            if not np.all(np.isclose(p_sum_, 1.0)):
                warnings.warn(
                    f"`p` parameters sum to {p_sum_}, instead of 1.0. "
                    "They will be automatically rescaled. "
                    "You can rescale them directly to get rid of this warning.",
                    UserWarning,
                )
                p_ = p_ / pt.sum(p_, axis=-1, keepdims=True)
                p = pt.as_tensor_variable(p_)
        return super().dist([p], **kwargs)

    def support_point(rv, size, p):
        mode = pt.argmax(p, axis=-1)
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode

    def logp(value, p):
        k = pt.shape(p)[-1]
        value_clip = pt.clip(value, 0, k - 1)

        # In the standard case p has one more dimension than value
        dim_diff = p.type.ndim - value.type.ndim
        if dim_diff > 1:
            # p brodacasts implicitly beyond value
            value_clip = pt.shape_padleft(value_clip, dim_diff - 1)
        elif dim_diff < 1:
            # value broadcasts implicitly beyond p
            p = pt.shape_padleft(p, 1 - dim_diff)

        a = pt.log(pt.take_along_axis(p, value_clip[..., None], axis=-1).squeeze(-1))

        res = pt.switch(
            pt.or_(pt.lt(value, 0), pt.gt(value, k - 1)),
            -np.inf,
            a,
        )

        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            pt.isclose(pt.sum(p, axis=-1), 1),
            msg="0 <= p <=1, sum(p) = 1",
        )

class OrderedLogistic:
    R"""Ordered Logistic distribution.

    Useful for regression on ordinal data values whose values range
    from 1 to K as a function of some predictor, :math:`\eta`. The
    cutpoints, :math:`c`, separate which ranges of :math:`\eta` are
    mapped to which of the K observed dependent variables. The number
    of cutpoints is K - 1. It is recommended that the cutpoints are
    constrained to be ordered.

    .. math::

       f(k \mid \eta, c) = \left\{
         \begin{array}{l}
           1 - \text{logit}^{-1}(\eta - c_1)
             \,, \text{if } k = 0 \\
           \text{logit}^{-1}(\eta - c_{k - 1}) -
           \text{logit}^{-1}(\eta - c_{k})
             \,, \text{if } 0 < k < K \\
           \text{logit}^{-1}(\eta - c_{K - 1})
             \,, \text{if } k = K \\
         \end{array}
       \right.

    Parameters
    ----------
    eta : tensor_like of float
        The predictor.
    cutpoints : tensor_like of array
        The length K - 1 array of cutpoints which break :math:`\eta` into
        ranges. Do not explicitly set the first and last elements of
        :math:`c` to negative and positive infinity.
    compute_p: boolean, default True
        Whether to compute and store in the trace the inferred probabilities of each categories,
        based on the cutpoints' values. Defaults to True.
        Might be useful to disable it if memory usage is of interest.

    Examples
    --------
    .. code-block:: python

        # Generate data for a simple 1 dimensional example problem
        n1_c = 300; n2_c = 300; n3_c = 300
        cluster1 = np.random.randn(n1_c) + -1
        cluster2 = np.random.randn(n2_c) + 0
        cluster3 = np.random.randn(n3_c) + 2

        x = np.concatenate((cluster1, cluster2, cluster3))
        y = np.concatenate((1*np.ones(n1_c),
                            2*np.ones(n2_c),
                            3*np.ones(n3_c))) - 1

        # Ordered logistic regression
        with pm.Model() as model:
            cutpoints = pm.Normal("cutpoints", mu=[-1,1], sigma=10, shape=2,
                                  transform=pm.distributions.transforms.ordered)
            y_ = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=x, observed=y)
            idata = pm.sample()

        # Plot the results
        plt.hist(cluster1, 30, alpha=0.5);
        plt.hist(cluster2, 30, alpha=0.5);
        plt.hist(cluster3, 30, alpha=0.5);
        posterior = idata.posterior.stack(sample=("chain", "draw"))
        plt.hist(posterior["cutpoints"][0], 80, alpha=0.2, color='k');
        plt.hist(posterior["cutpoints"][1], 80, alpha=0.2, color='k');
    """

    def __new__(cls, name, eta, cutpoints, compute_p=True, **kwargs):
        p = cls.compute_p(eta, cutpoints)
        if compute_p:
            p = pm.Deterministic(f"{name}_probs", p)
        out_rv = Categorical(name, p=p, **kwargs)
        return out_rv

    @classmethod
    def dist(cls, eta, cutpoints, **kwargs):
        p = cls.compute_p(eta, cutpoints)
        return Categorical.dist(p=p, **kwargs)

    @classmethod
    def compute_p(cls, eta, cutpoints):
        eta = pt.as_tensor_variable(eta)
        cutpoints = pt.as_tensor_variable(cutpoints)

        pa = sigmoid(cutpoints - pt.shape_padright(eta))
        p_cum = pt.concatenate(
            [
                pt.zeros_like(pt.shape_padright(pa[..., 0])),
                pa,
                pt.ones_like(pt.shape_padright(pa[..., 0])),
            ],
            axis=-1,
        )
        p = p_cum[..., 1:] - p_cum[..., :-1]
        return p
