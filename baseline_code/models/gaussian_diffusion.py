"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
from tqdm.auto import tqdm

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood

from PIL import Image
from baseline_code.utils import logger

import os

import csv

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
	"""
    生成一个加噪方案：线性or非线性
    Get a pre-defined beta schedule for the given name.
    获取给定名称的预定义测试时间表（num_diffusion_timesteps默认1000）

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
	if schedule_name == "linear":
		# Linear schedule from Ho et al, extended to work for any number of
		# diffusion steps. 原始DDPM用的是线性加噪方案(线性增长的过程)
		scale = 1000 / num_diffusion_timesteps  # scale=1.0
		beta_start = scale * 0.0001
		beta_end = scale * 0.02
		return np.linspace(  # 生成1000个均匀分布的数值序列array([*,*,...,*])
			beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
		)
	elif schedule_name == "cosine":  # IDDPM用的是余弦加噪方案
		return betas_for_alpha_bar(
			num_diffusion_timesteps,
			lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
		)
	else:
		raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
	"""
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
	betas = []
	for i in range(num_diffusion_timesteps):
		t1 = i / num_diffusion_timesteps
		t2 = (i + 1) / num_diffusion_timesteps
		betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))  # 控制beta上界，最大为0.999
	return np.array(betas)


class ModelMeanType(enum.Enum):
	"""
    Which type of output the model predicts.
    """

	PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
	START_X = enum.auto()  # the model predicts x_0
	EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
	"""
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

	LEARNED = enum.auto()  # 以允许模型进行预测()之间的值
	FIXED_SMALL = enum.auto()
	FIXED_LARGE = enum.auto()
	LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
	MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
	RESCALED_MSE = (
		enum.auto()
	)  # use raw MSE loss (with RESCALED_KL when learning variances)
	KL = enum.auto()  # use the variational lower-bound
	RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

	def is_vb(self):
		return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
	"""
    Utilities for training and sampling diffusion models.
    用于训练和采样扩散模型的实用程序。
    直接从这里移植，然后随着时间的推移进行进一步的实验
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

	def __init__(
			self,
			*,
			betas,
			model_mean_type,
			model_var_type,
			loss_type,
			rescale_timesteps=False,
			conditioning_noise="constant",
	):
		self.model_mean_type = model_mean_type  # 模型预测噪声，还是均值，还是x0
		self.model_var_type = model_var_type    # = gd.ModelVarType.FIXED_LARGE
		# 模型的方差类型，可学习，还是固定，固定beta，还是beta_bar
		self.loss_type = loss_type              # 此处使用最普通的MSE
		self.rescale_timesteps = rescale_timesteps  # False
		# 如果为True，则将浮点时间步长传入模型，以便它们总是按比例缩放；原始论文(0 ~ 1000)
		self.conditioning_noise = conditioning_noise
		assert self.conditioning_noise in ["reverse", "constant"]

		# Use float64 for accuracy.
		# 原始的betas
		betas = np.array(betas, dtype=np.float64)  # 0-1之间的一维向量
		self.betas = betas
		assert len(betas.shape) == 1, "betas must be 1-D"
		assert (betas > 0).all() and (betas <= 1).all()

		# 根据betas的长度来确定的num_timesteps，betas是已经修改过的，故此处num_timesteps是新的时间序列的长度
		self.num_timesteps = int(betas.shape[0])

		alphas = 1.0 - betas
		self.alphas_cumprod = np.cumprod(alphas, axis=0)  # 计算alphas_t_bar
		# 计算alphas_t-1_bar(好计算，不包含alphas_t_bar最后一项即[:-1]，第0项用1填充即可)
		self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
		# 计算alphas_t+1_bar(从alphas_t_bar第一项开始传入，最后一项用0填充
		self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
		assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

		# calculations for diffusion q(x_t | x_{t-1}) and others
		self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)  # sqrt下 alphas_t_bar
		self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)  # sqrt下 1-alphas_t_bar
		self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)  # log(1-alphas_t_bar),一般和求误差函数有关
		self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)  # sqrt下 1/alphas_t_bar
		self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)  # sqrt下 1/alphas_t_bar -1,算后验分布时会用到

		# calculations for posterior q(x_{t-1} | x_t, x_0)
		self.posterior_variance = (  # 后验分布的真实方差betas_bar，是个常数
				betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		# log calculation clipped because the posterior variance is 0 at the
		# beginning of the diffusion chain.
		self.posterior_log_variance_clipped = np.log(  # 对betas_bar取log，并进行截断(防止第1项变成0)
			np.append(self.posterior_variance[1], self.posterior_variance[1:])
		)
		self.posterior_mean_coef1 = (  # 后验均值ut_bar(xt,x0)的第1个系数
				betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_mean_coef2 = (  # 后验均值ut_bar(xt,x0)的第2个系数
				(1.0 - self.alphas_cumprod_prev)
				* np.sqrt(alphas)
				/ (1.0 - self.alphas_cumprod)
		)

	def q_mean_variance(self, x_start, t):
		"""
        Get the distribution q(x_t | x_0).  q--真实分布，基于x0和t，计算出xt(IDDPM公式8，不含参)

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
		mean = (
				_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
		)
		# _extract_into_tensor函数作用：把张量sqrt_alphas_cumprod的第t个取出来，且形状要等于x_start.shape
		variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
		log_variance = _extract_into_tensor(
			self.log_one_minus_alphas_cumprod, t, x_start.shape
		)
		return mean, variance, log_variance  # 返回均值、方差、对数方差

	def q_sample(self, x_start, t, noise=None):  # 给定x0和t，采样出xt, 相当于重参数对q_mean_variance进行采样
		"""
        对公式8进行重参数的过程，即前向过程
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
		if noise is None:
			noise = th.randn_like(x_start)
			# 重参数化第1步，生成一个标准噪声；
			# randn_like即要与x_start生成一样大小的量；randn是normal的意思，即从标准分布中生成
		assert noise.shape == x_start.shape
		return (
				_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
				+ _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
				* noise
		)

	# 重参数采样过程，noise * 标准差 + 均值

	def q_sample_conditioning(self, model_kwargs, t, train: bool = True):

		# perturb conditioning image with noise as in the forward diffusion process
		'''if train:
            if self.conditioning_noise == 'reverse':
                t_reverse_diffusion = (self.num_timesteps - 1 - t).to(device=dist_util.dev())
                conditioning_x = self.q_sample(model_kwargs["image"], t_reverse_diffusion, noise=None)
            elif self.conditioning_noise == 'constant':
                conditioning_x = model_kwargs["image"] + th.randn_like(model_kwargs["image"])
        else:'''

		conditioning_x = model_kwargs["image"]

		model_kwargs["conditioning_x"] = conditioning_x

	def q_posterior_mean_variance(self, x_start, x_t, t):
		"""
        Compute the mean and variance of the diffusion posterior:
            计算IDDPM中的公式9和10，即后验分布真实的均值和方差
            q(x_{t-1} | x_t, x_0)

        """
		assert x_start.shape == x_t.shape
		# 后验分布的均值
		posterior_mean = (
				_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
				+ _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
		)
		# 后验分布的方差，之前计算过了，直接取出来就行
		posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
		# 后验分布的对数方差，之前计算过了，直接取出来就行
		posterior_log_variance_clipped = _extract_into_tensor(
			self.posterior_log_variance_clipped, t, x_t.shape
		)
		# assert一下，第0个形状大小必须一样；第0个形状其实就是bath_size
		assert (
				posterior_mean.shape[0]
				== posterior_variance.shape[0]
				== posterior_log_variance_clipped.shape[0]
				== x_start.shape[0]
		)
		return posterior_mean, posterior_variance, posterior_log_variance_clipped

	def p_mean_variance(
			self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
	):
		"""
        这个函数很重要！！！
        p分布---NN预测的分布，得到逆扩散过程前一时刻的均值&方差，从xt预测xt-1
        也包括初始数据分布x_start的预测； model其实就是unet中的model,没有x_start，因为希望用model预测/拟合出x[t-1]的均值和方差

        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
		if model_kwargs is None:
			model_kwargs = {}

		B, C = x.shape[:2]
		assert t.shape == (B,)
		# model接受当前时刻分布的采样值x和当前时刻t作为输入，
		model_output = model(x, self._scale_timesteps(t), **model_kwargs)

		# 得到方差和对数方差
		if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
			# 可学习的方差
			# C * 2，即通道数变成2倍的原因：既要预测均值，又要预测方差
			assert model_output.shape == (B, C * 2, *x.shape[2:])
			# 对2倍通道进行第1维(通道)split分割，分割出model_output和model_var_values(与方差有关)
			model_output, model_var_values = th.split(model_output, C, dim=1)
			if self.model_var_type == ModelVarType.LEARNED:
				# 直接预测方差
				model_log_variance = model_var_values  # 预测的是对数方差，从x[t]到x[t-1]的对数方差
				model_variance = th.exp(model_log_variance)
			else:
				# 预测方差差值的系数/范围，公式14
				# 预测的范围是[-1，1]之间
				# beta_t_bar (beta_t_bar < beta_t)
				min_log = _extract_into_tensor(
					self.posterior_log_variance_clipped, t, x.shape
				)
				# beta_t
				max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
				# The model_var_values is [-1, 1] for [min_var, max_var].
				# 将[-1, 1]之间转化成[0, 1]之间
				frac = (model_var_values + 1) / 2
				model_log_variance = frac * max_log + (1 - frac) * min_log  # 公式14
				model_variance = th.exp(model_log_variance)  # 最后取指数操作
		else:
			# 不可学习的方差(固定的方差)
			model_variance, model_log_variance = {
				# for fixedlarge, we set the initial (log-)variance like so
				# to get a better decoder log likelihood.
				ModelVarType.FIXED_LARGE: (  # 直接用beta_t；': '表示字典形式
					# 初始值用variance比较好，剩余的用固定的beta_t
					np.append(self.posterior_variance[1], self.betas[1:]),
					np.log(np.append(self.posterior_variance[1], self.betas[1:])),
				),
				ModelVarType.FIXED_SMALL: (  # 用beta_t_bar
					self.posterior_variance,
					self.posterior_log_variance_clipped,
				),
			}[self.model_var_type]
			# [self.model_var_type]表示将model_var_type传入这2个字典中，
			# 如果self.model_var_type是FIXED_LARGE,得到的就是上面直接用beta_t的值，
			# 如果是FIXED_SMALL，得到的就是用beta_t_bar的值

			# 上面计算的是所有时刻的，现在再取出第t时刻的方差和对数当差
			model_variance = _extract_into_tensor(model_variance, t, x.shape)
			model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

		def process_xstart(x):
			""" 对每一步的输出进行一定的后处理"""
			if denoised_fn is not None:
				x = denoised_fn(x, **model_kwargs)
			if clip_denoised:
				return x.clamp(-1, 1)
			return x

		# 对NN预测的是噪声、期望均值还是x0进行一系列判断
		if self.model_mean_type == ModelMeanType.PREVIOUS_X:
			# case1：预测x[t-1]的期望值,即u_t_bar
			# pred_xstart在训练中用不到，评估中可以用到
			pred_xstart = process_xstart(
				self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
			)
			model_mean = model_output
		elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
			if self.model_mean_type == ModelMeanType.START_X:
				# case2：预测x[0]的期望值
				pred_xstart = process_xstart(model_output)
			else:
				# case3：预测eps的期望值
				pred_xstart = process_xstart(
					self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
					# 辅助函数，从xt中预测xo，公式12
				)
				# pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=process_xstart(model_output)) didnt work

			model_mean, _, _ = self.q_posterior_mean_variance(
				x_start=pred_xstart, x_t=x, t=t
			)
		else:
			raise NotImplementedError(self.model_mean_type)

		assert (
				model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
		)
		return {  # 返回前一时刻的均值和方差
			"mean": model_mean,
			"variance": model_variance,
			"log_variance": model_log_variance,
			"pred_xstart": pred_xstart,
		}

	def _predict_xstart_from_eps(self, x_t, t, eps):
		# 辅助函数，从xt中预测xo，公式12
		assert x_t.shape == eps.shape
		return (
				_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
				- _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
		)

	def _predict_xstart_from_xprev(self, x_t, t, xprev):
		# 辅助函数，从xt-1中预测xo，公式10
		assert x_t.shape == xprev.shape
		return (  # (xprev - coef2*x_t) / coef1
				_extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
				- _extract_into_tensor(
			self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
		)
				* x_t
		)

	def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
		# 从x0预测eps，公式8反推， 从x0和xt反推出所加的噪声是多少
		return (
				       _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
				       - pred_xstart
		       ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

	def _scale_timesteps(self, t):
		if self.rescale_timesteps:
			return t.float() * (1000.0 / self.num_timesteps)
		return t

	def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
		"""
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
		gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
		new_mean = (
				p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
		)
		return new_mean

	def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
		"""
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
		alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
		eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
		class_cond = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
		# class_cond = cond_fn(model_kwargs['image'], self._scale_timesteps(t), **model_kwargs)
		eps = eps - (1 - alpha_bar).sqrt() * class_cond
		out = p_mean_var.copy()
		out["class_cond"] = class_cond
		out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
		out["mean"], _, _ = self.q_posterior_mean_variance(
			x_start=out["pred_xstart"], x_t=x, t=t
		)
		return out

	def p_sample(
			self,
			model,
			x,
			t,
			clip_denoised=True,
			denoised_fn=None,
			cond_fn=None,
			model_kwargs=None,
	):
		"""
        基于x[t]采样出x[t-1],相当于噪声的恢复，
        得到x[t-1]的均值、方差、对数方差、x[0]的预测值
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
		# 得到x[t-1]的均值、方差、对数方差、x[0]的预测值
		out = self.p_mean_variance(
			model,
			x,
			t,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
		)
		noise = th.randn_like(x)
		# 非零时刻的掩码矩阵
		nonzero_mask = (
			(t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
		)  # no noise when t == 0
		if cond_fn is not None:
			out["mean"] = self.condition_mean(
				cond_fn, out, x, t, model_kwargs=model_kwargs
			)
		# 重采样出t-1时刻的sample：noise * 标准差
		sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
		return {"sample": sample, "pred_xstart": out["pred_xstart"]}

	def p_sample_loop(
			self,
			model,
			shape,
			noise=None,
			clip_denoised=True,
			denoised_fn=None,
			cond_fn=None,
			model_kwargs=None,
			device=None,
			progress=False,
	):
		"""
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
		final = None
		for sample in self.p_sample_loop_progressive(
				model,
				shape,
				noise=noise,
				clip_denoised=clip_denoised,
				denoised_fn=denoised_fn,
				cond_fn=cond_fn,
				model_kwargs=model_kwargs,
				device=device,
				progress=progress,
		):
			final = sample
		return final["sample"]

	def p_sample_loop_progressive(
			self,
			model,
			shape,
			noise=None,
			clip_denoised=True,
			denoised_fn=None,
			cond_fn=None,
			model_kwargs=None,
			device=None,
			progress=False,
	):
		"""
        递进式地进行循环采样
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
		if device is None:
			device = next(model.parameters()).device
		assert isinstance(shape, (tuple, list))
		# 首先生成一个noise,即T时刻的标准噪音
		if noise is not None:
			img = noise
		else:
			img = th.randn(*shape, device=device)

		# 对时间进行倒序索引,'::'符号代表对列表进行逆序，从T倒推到0
		indices = list(range(self.num_timesteps))[::-1]

		if progress:
			# Lazy import so that we don't depend on tqdm.
			from tqdm.auto import tqdm

			indices = tqdm(indices)

		# 对倒序后的时间索引indices进行遍历，标准推理过程推理出x0，不需要算梯度
		image = []
		for step, i in enumerate(indices):
			t = th.tensor([i] * shape[0], device=device)

			with th.no_grad():  # 不需要算梯度，加上th.no_grad()更高效
				out = self.p_sample(
					model,
					img,
					t,
					clip_denoised=clip_denoised,
					denoised_fn=denoised_fn,
					cond_fn=cond_fn,
					model_kwargs=model_kwargs,
				)
				# if (step + 1) % 5 == 0:
				# 	print('step', step)
				# 	samples = out["sample"]  # [1 3 256 256] double
				# 	print(type(samples))
				# 	print(samples.size())
				# 	print('111111')
				# 	samples = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8)  # [1 3 256 256] unit8
				# 	print(type(samples))
				# 	print(samples.size())
				# 	print('222222')
				# 	samples = samples.permute(0, 2, 3, 1)  # # [1 256 256 3] unit8
				# 	print(type(samples))
				# 	print(samples.size())
				# 	print('333333')
				# 	samples = samples.contiguous()  # [1 256 256 3] unit8
				# 	print(type(samples))
				# 	print(samples.size())
				# 	print('++++++')
				#
				# 	image.extend([sample.cpu().numpy() for sample in [samples]])  #
				# 	print(type(image))
				# 	print(image.size())
				# 	print('++++++')
				#
				# 	arr = np.concatenate(image, axis=0)
				# 	img = Image.fromarray(arr[-1])
				#
				# 	out_path = os.path.join(logger.get_dir(), f"/chenxue/experiment_data/anomaly_diffusion_healthy_only_train_brats2021_64x64/samples_label_{str(step).zfill(4)}.npz")
				# 	out_image = os.path.join(logger.get_dir(), f"/chenxue/experiment_data/anomaly_diffusion_healthy_only_train_brats2021_64x64/samples_label_{str(step).zfill(4)}.tif")
				# 	print('out_npz_path:{}'.format(out_path))
				# 	print('out_tif_path:{}'.format(out_image))
				# 	img.save(out_image, compression='raw')
				# 	np.savez(out_path, arr[-1])
				yield out
				img = out["sample"]

	def ddim_sample(  # 单步时刻t的采样
			self,
			model,
			x,  # 输入x即Xt,最终输出Xt-1
			t,
			clip_denoised=True,
			denoised_fn=None,
			cond_fn=None,
			model_kwargs=None,
			eta=0.0,
	):
		"""
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
		out = self.p_mean_variance(  # 计算前一时刻的均值和方差,得到的out是一个字典
			model,
			x,
			t,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
		)
		score_mean = out["mean"]
		# print('*******ddim_sample*******')
		# print(out)      # 'mean' 'variance' 'log_variance' 'pred_xstart'
		# print('*******ddim_sample*******')
		# if (t != self.num_timesteps - 1).all():
		#    cond_fn = None
		if cond_fn is not None:
			out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

		# Usually our model outputs epsilon, but we re-derive it
		# in case we used x_start or x_prev prediction.
		eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])  # 基于重采样公式计算epsion

		alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
		alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
		sigma = (  # sigma_t
				eta
				* th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
				* th.sqrt(1 - alpha_bar / alpha_bar_prev)
		)
		# Equation 12.
		noise = th.randn_like(x)
		mean_pred = (  # 重采样均值mean_pred
				out["pred_xstart"] * th.sqrt(alpha_bar_prev)
				+ th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
		)
		nonzero_mask = (  # 当t!=0时,需要有noise这一项;当t==0时,不需要noise,直接输出均值
			(t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
		)  # no noise when t == 0
		sample = mean_pred + nonzero_mask * sigma * noise  # 重采样得到的sample
		result_dict = {"sample": sample, "pred_xstart": out["pred_xstart"], "score_mean": score_mean}
		if cond_fn is not None:
			result_dict.update({"class_cond": out["class_cond"]})

		return result_dict

	def ddim_reverse_sample(
			self,
			model,
			x,
			t,
			clip_denoised=True,
			denoised_fn=None,
			model_kwargs=None,
			eta=0.0,
	):
		"""
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
		assert eta == 0.0, "Reverse ODE only for deterministic path"
		# 返回前一时刻的均值和方差 "mean" "variance" "log_variance" "pred_xstart"
		out = self.p_mean_variance(
			model,
			x,
			t,
			clip_denoised=clip_denoised,
			denoised_fn=denoised_fn,
			model_kwargs=model_kwargs,
		)
		# Usually our model outputs epsilon, but we re-derive it
		# in case we used x_start or x_prev prediction.
		# 通常我们的模型输出epsilon，但如果我们使用x_start或x_prev预测，我们将重新推导它。
		eps = (
				      _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
				      - out["pred_xstart"]
		      ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
		alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

		# Equation 12. reversed
		mean_pred = (
				out["pred_xstart"] * th.sqrt(alpha_bar_next)
				+ th.sqrt(1 - alpha_bar_next) * eps
		)

		return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

	def ddim_sample_loop(  # 完整的采样过程,在单步采样ddim_sample外面加一层for循环
			self,
			model,
			shape,
			noise=None,
			clip_denoised=True,
			denoised_fn=None,
			cond_fn=None,
			model_kwargs=None,
			device=None,
			progress=False,
			reconstruction: bool = False,
			eta=0.0,
			sampling_progression_ratio=1.0
	):
		"""
        Generate samples from the model using DDIM.
        使用DDIM从模型生成样本

        Same usage as p_sample_loop().
        """

		final = [] if progress else None

		flag_number = 1
		# 采样74次
		for sample in self.ddim_sample_loop_progressive(
				model,
				shape,
				noise=noise,
				clip_denoised=clip_denoised,
				denoised_fn=denoised_fn,
				cond_fn=cond_fn,
				model_kwargs=model_kwargs,
				device=device,
				eta=eta,
				reverse=reconstruction,
				sampling_progression_ratio=sampling_progression_ratio
		):
			print("flag_number:{}".format(flag_number))
			flag_number += 1
			if progress:
				final.append(sample)
			else:
				final = sample
		final_output = "sample"  # if denoised_fn is None else "pred_xstart"
		return (final[-1][final_output], final) if progress else (final[final_output], [final])

	def ddim_sample_loop_progressive(
			self,
			model,
			shape,
			noise=None,
			clip_denoised=True,
			denoised_fn=None,
			cond_fn=None,
			model_kwargs=None,
			device=None,
			eta=0.0,
			reverse: bool = False,
			sampling_progression_ratio: float = 1
	):
		"""
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.  利用DDIM对模型进行采样，从DDIM的每个时间步长得到中间样本。

        Same usage as p_sample_loop_progressive().
        """
		if device is None:
			device = next(model.parameters()).device
		assert isinstance(shape, (tuple, list))

		indices = list(range(self.num_timesteps))  # 0 -> T num_timesteps是新的时间序列的长度，其实就是更新后的beta序列
		# print('num_timesteps:{} '.format(self.num_timesteps))       # 100
		# print('indices: {}'.format(indices))
		# start diffusion from a particular point or run it only up to a point
		indices = indices[:int(len(indices) * sampling_progression_ratio)]  # 0 -> L*T
		# print('indices(0 -> L*T): {}'.format(indices))  # indices从100变成了74

		if not reverse:  # not reverse == decoding, reverse == encoding
			indices = indices[::-1]  # L*T -> 0
			assert noise is not None, "Reverse DDIM requires input noise as an image"

		if noise is not None:
			img = noise
		else:
			img = th.randn(*shape, device=device)
		print('type(shape):{}'.format(type(shape)))         # <class 'tuple'>
		print('shape:{}'.format(shape))                # 4 -->shape:(64, 4, 64, 64)
		# ^^^^^^^^^^^^^^^^^^^^^^^^^^
		# split:test, patient_id:2, slice_id:42
		# ^^^^^^^^^^^^^^^^^^^^^^^^^^
		print("model_kwargs")
		print(model_kwargs)
		print("model_kwargs finished")
		indices = tqdm(indices)         # 进度条，返回一个可迭代对象

		image = []
		# 按理说indices从100变成了74： indices(0 -> L*T): [0, 1, 2, 3, 4, 5, 6, ..., 71, 72, 73, 74]
		for step, i in enumerate(indices):  # 遍历，将indices送入ddim_sample中，0<=step<=74
			print('step={},i={}'.format(step, i))       # step=1...74, i=1...74
			t = th.tensor([i] * shape[0], device=device)
			# print("t:{}".format(t))
			# t:tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			#         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
			# t:tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
			#         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
			with th.no_grad():
				if reverse:         # reverse == encoding
					out = self.ddim_reverse_sample(
						model,
						img,
						t,
						clip_denoised=clip_denoised,
						denoised_fn=denoised_fn,
						model_kwargs=model_kwargs,
						eta=0.0,
					)
					# 保存ecoding当前step每一张sample涂（共batchsize张）

					# 保存ecoding当前step每一张pred_xstart图（共batchsize张）
				else:               # not reverse == decoding
					out = self.ddim_sample(
						model,
						img,
						t,
						clip_denoised=clip_denoised,
						denoised_fn=denoised_fn,
						cond_fn=cond_fn,
						model_kwargs=model_kwargs,
						eta=eta,
					)

				yield out
				img = out["sample"]
				calc_bpd = self.calc_bpd_loop(model, x_start=img, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
				# print('calc_bpd print:')
				# print(calc_bpd)
				# print('calc_bpd print finished')

	def _vb_terms_bpd(
			self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
	):
		"""
        计算loss,需要优化的KL散度；主要用到公式5和6
        vb：变分下界，bpd：每个维度的比特数，一般用来对不同论文做对比用
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
		# 真实的x[0]、x[t]和t去计算出x[t-1]的均值和方差
		true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
			x_start=x_start, x_t=x_t, t=t
		)
		# 计算过真实的均值和方差以后，用NN去预测均值和方差，
		# x[t]、t和预测的x0去计算出x[t-1]的均值和方差
		out = self.p_mean_variance(
			model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
		)
		# p_mean_variance() 返回：a dict with the following keys:
		#                  - 'mean': the model mean output.
		#                  - 'variance': the model variance output.
		#                  - 'log_variance': the log of 'variance'.
		#                  - 'pred_xstart': the prediction for x_0.
		# p_theta与q分布之间的KL散度，公式6
		# 对应着L[t-1]损失函数
		kl = normal_kl(  # 2个高斯分布之间的KL散度
			true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
		)
		kl = mean_flat(kl) / np.log(2.0)

		# 对应着L[0]损失函数，用的是负对数似然函数，累积分布函数的差分去拟合离散的高斯分布，公式5
		# 只针对于t=0时刻，即计算x1->x0的均值和方差，对应paper中的L0
		decoder_nll = -discretized_gaussian_log_likelihood(
			x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
		)
		assert decoder_nll.shape == x_start.shape
		decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

		# At the first timestep return the decoder NLL,
		# otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
		# t=0时刻，用离散的高斯分布去计算似然
		# t>0时刻，直接用KL散度
		output = th.where((t == 0), decoder_nll, kl)
		# print('<<<<<<output<<<<<<<<<<')
		# print(output)
		# print(output.shape)
		# print('<<<<<<<<<<<<<<<<')
		return {"output": output, "pred_xstart": out["pred_xstart"]}

	def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
		"""
        三种case：只学习vb loss；只学习MSE loss；同时学习MSE和vb loss
        Compute training losses for a single timestep.
        计算单个时间步长的训练损失。
        :param model: the model to evaluate loss on. U-Net模型，Xt作为模型输入，epsion方差作为模型输出
        :param x_start: the [N x C x ...] tensor of inputs. X0，即训练集
        :param t: a batch of timestep indices. 相当于embedding
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
		if model_kwargs is None:
			model_kwargs = {}

		if noise is None:   
			noise = th.randn_like(x_start)          # 预测的噪声
		# 基于x[0]和任意时刻t 以及 噪音采样出x[t]
		# 生成Xt的样本，根据X0(即x_start)和t，通过边缘分布的重参数化生成任意时刻Xt
		x_t = self.q_sample(x_start, t, noise=noise)    # 计算前向过程t时刻的噪音x[t]

		terms = {}

		# 如果损失函数用的是KL散度
		if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
			terms["loss"] = self._vb_terms_bpd(
				model=model,
				x_start=x_start,
				x_t=x_t,
				t=t,
				clip_denoised=False,
				model_kwargs=model_kwargs,
			)["output"]  # 字典，返回L0或Lt-1，取决于t是否为0，即这里公式(5)(6)合在一起表示了
			if self.loss_type == LossType.RESCALED_KL:
				terms["loss"] *= self.num_timesteps  # 如果采用rescaled操作优化，还需要乘权重num_timesteps

		# 如果loss函数类型是MSE(此处是MSE not KL)
		elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
			# 将上面生成的Xt和时刻t(即self._scale_timesteps(t))一起送入UNet模型model(,,**)中

			""" !!!!!!model预测反向去噪过程t时刻的噪音!!!!!! """
			model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
			# UNet模型的输出model_output一般情况是预测的方差espion(有的人也喜欢预测X0,or均值or期望值等)

			if self.model_var_type in [  # 如果方差可学习,走这一步.否则直接走target = {...}的代码
				ModelVarType.LEARNED,
				ModelVarType.LEARNED_RANGE,     # 此处self.model_var_type = gd.ModelVarType.FIXED_LARGE
			]:
				B, C = x_t.shape[:2]
				assert model_output.shape == (B, C * 2, *x_t.shape[2:])
				model_output, model_var_values = th.split(model_output, C, dim=1)
				# Learn the variance using the variational bound, but don't let
				# it affect our mean prediction.
				# 将model_output和方差拼起来
				frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
				terms["vb"] = self._vb_terms_bpd(
					model=lambda *args, r=frozen_out: r,  # 不需要再传入model了，让学到的方差不影响均值的预测
					x_start=x_start,
					x_t=x_t,
					t=t,
					clip_denoised=False,
				)["output"]
				if self.loss_type == LossType.RESCALED_MSE:
					# Divide by 1000 for equivalence with initial implementation.
					# Without a factor of 1/1000, the VB term hurts the MSE term.
					terms["vb"] *= self.num_timesteps / 1000.0

			target = {  # 字典,返回UNet模型预测对应的目标值，即真实值
				# 此处NN预测的是方差：gd.ModelMeanType.EPSILON，故返回noise
				ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
					x_start=x_start, x_t=x_t, t=t
				)[0],  # 如果预测前一时刻均值PREVIOUS_X,键值返回q_posterior_mean_variance(...)[0]，[0]项就是均值
				ModelMeanType.START_X: x_start,     # 如果预测X0,返回x_start
				ModelMeanType.EPSILON: noise,       # 如果预测X0,返回noise(一般都是预测noise)
			}[self.model_mean_type]                 # self.model_mean_type = gd.ModelMeanType.EPSILON
			assert model_output.shape == target.shape == x_start.shape
			terms["mse"] = mean_flat(
				(target - model_output) ** 2)  # loss的计算就很简单了,即Lsimple,直接计算(target - model_output)的平方差
			if "vb" in terms:
				terms["loss"] = terms["mse"] + terms["vb"]
				# print('loss = mse + vb')
			else:
				terms["loss"] = terms["mse"]
				# print('loss = mse')
		else:
			raise NotImplementedError(self.loss_type)
		# print(':{}'.format(terms))
		# print('***')
		return terms

	def _prior_bpd(self, x_start):
		"""
        先验的KL散度，不影响模型训练，公式7(不含参)
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
		batch_size = x_start.shape[0]
		t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
		qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
		kl_prior = normal_kl(
			mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
		)
		return mean_flat(kl_prior) / np.log(2.0)

	def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
		"""
        从T时刻到0时刻，把所有的loss都计算出来；
        不会在训练中用到，只是用来做评估的
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
		device = x_start.device
		batch_size = x_start.shape[0]

		vb = []
		xstart_mse = []
		mse = []
		for t in list(range(self.num_timesteps))[::-1]:
			t_batch = th.tensor([t] * batch_size, device=device)
			noise = th.randn_like(x_start)
			x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
			# Calculate VLB term at the current timestep
			with th.no_grad():
				out = self._vb_terms_bpd(
					model,
					x_start=x_start,
					x_t=x_t,
					t=t_batch,
					clip_denoised=clip_denoised,
					model_kwargs=model_kwargs,
				)
			vb.append(out["output"])
			xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
			eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
			mse.append(mean_flat((eps - noise) ** 2))

		vb = th.stack(vb, dim=1)
		xstart_mse = th.stack(xstart_mse, dim=1)
		mse = th.stack(mse, dim=1)

		prior_bpd = self._prior_bpd(x_start)    # 先验KL散度，与训练无关，只有ecoder决定
		total_bpd = vb.sum(dim=1) + prior_bpd
		# print('*****calc_bpd_loop print******')
		# print("total_bpd:{}".format(total_bpd))
		# print("prior_bpd:{}".format(prior_bpd))
		# print("vb:{}".format(vb))                   # 目测1-10以内
		# print("xstart_mse:{}".format(xstart_mse))   # 目测0-2以内，求平均
		# print("mse:{}".format(mse))     # 目测1-10以内，求平均
		# print('*****calc_bpd_loop print finished******')
		print('*****out print******')
		mse_per_batch = 0.0
		num = 0
		numpy_vb = vb.cpu().numpy().float()
		numpy_start_mse = xstart_mse.cpu().numpy().float()
		numpy_mse = mse.cpu().numpy().float()
		# print('shape_mse:{}'.format(numpy_mse))
		# [r, c] = shape_mse.shape
		# print('r:{}, c:{}'.format(r, c))
		# for i in range(r):
		# 	for j in range(c):
		# 		mse_per_batch += shape_mse[i, j]
		# 		num += 1
		# print('mse_per_batch:{}'.format(mse_per_batch))
		# print('num:{}'.format(num))
		# mse_per_batch = mse_per_batch / (num * 1.0)
		vb_per_batch = numpy_vb.mean().item
		start_mse_per_batch = numpy_start_mse.mean().item
		mse_per_batch = numpy_mse.mean().item
		row = []
		row.append(float(vb_per_batch))
		row.append(float(start_mse_per_batch))
		row.append(float(mse_per_batch))
		with open('/chenxue/paper3/test/mse_per_batch.csv', 'a', newline='') as csvfile:
			writer = csv.writer(csvfile)
			# writer = csv.writer(csvfile, delimiter='', quotechar='|', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(row)
		csvfile.close()
		print('mse_per_batch:{}'.format(mse_per_batch))
		print('*****out print finished******')
		return {
			"total_bpd": total_bpd,
			"prior_bpd": prior_bpd,
			"vb": vb,
			"xstart_mse": xstart_mse,
			"mse": mse,
		}


def _extract_into_tensor(arr, timesteps, broadcast_shape):
	"""
    辅助函数，从tensor中取第几个时刻
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
	# res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
	res = th.from_numpy(arr)
	_dev = timesteps.device
	res = res.to(device=_dev)
	res = res[timesteps].float()

	while len(res.shape) < len(broadcast_shape):
		res = res[..., None]
	return res.expand(broadcast_shape)


import torch
import torch.nn.functional as F
import random


def add_noise(x):
	# I use 16x16 to generate noise for 128x128 original resolution.
	ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], 16, 16)).to(x.device)
	res = F.upsample_bilinear(ns, size=(128, 128))
	# Smoothing after makes the noise look nicer/more natural but might not be necessary
	res = F.avg_pool2d(res, kernel_size=9, stride=1, padding=4)
	# Roll randomly so the distribution of the noise "centers" is not so regular after upsamling.
	# Also might be unnecessary.
	roll_x = random.choice(range(128))
	roll_y = random.choice(range(128))
	ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
	# Normalise to whatever std you want, 0.1 works well for me.
	# Also shouldn't do .mean .std across
	# the batch if applying noise to the whole batch.
	ns = (ns - ns.mean()) / ns.std() * 0.1
	# I only apply the noise in the foreground (brain) since adding noise
	# to the black background is detrimental to the model I think.
	mask = x.sum(dim=1, keepdim=True) > 0
	ns *= mask
	return x + ns