"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].即观测的数据x_start
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    将观测数据x_start带入到分布(用均值means和对数方差log_scales表示分布)中去，看他的可能性多大，这就是似然
    """
    assert x.shape == means.shape == log_scales.shape
    # 首先对观测数据减去均值
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)

    # 将[-1,1]分成255个bins，最右边的CDF记为1，最左边的CDF记为0
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)     # 得到plus_in(即右边)近似的标准分布
    cdf_plus = approx_standard_normal_cdf(plus_in)      # 计算出近似标准分布的累计分布函数

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)      # 得到min_in(即左边)近似的标准分布
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    # 用小范围的CDF之差来表示PDF
    cdf_delta = cdf_plus - cdf_min
    # 考虑到2个极限的地方，这里用到了2个where，最终得到对数概率
    log_probs = th.where(
        x < -0.999,     # 当x位于最左边，将其赋予0，则插值=右边-左边=log_cdf_plus-0=log_cdf_plus
        log_cdf_plus,   # x位于最左边时的插值
        # 当x位于最右边，将其赋予1，则插值=右边-左边=1-cdf_min=log_one_minus_cdf_min
        # 其余正常情况下，对cdf_delta取对数即可，即下式最后一项th.log(cdf_delta.clamp(min=1e-12))
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    # 最后返回对数似然
    return log_probs