import copy
import functools
import os
from random import randint
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torchvision
from typing import Dict
import matplotlib.pyplot as plt

from baseline_code.utils import dist_util, logger
from baseline_code.utils.fp16_util import MixedPrecisionTrainer
from baseline_code.models.nn import update_ema
from baseline_code.models.resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
	def __init__(
			self,
			*,
			model,
			diffusion,
			data,
			data_val,
			batch_size,
			microbatch,         # -1
			lr,
			ema_rate,
			log_interval,
			save_interval,
			resume_checkpoint,
			use_fp16=False,
			fp16_scale_growth=1e-3,
			schedule_sampler=None,
			weight_decay=0.0,
			lr_anneal_steps=0,
			main_data_indentifier: str = "image",
			# cond_dropout_rate: float = 0.0,
			cond_dropout_rate: float = 1.0,
			conditioning_variable: str = "y",
			iterations: int = 30e3,
	):
		self.model = model
		self.diffusion = diffusion
		self.data = data
		self.data_val = data_val
		self.batch_size = batch_size
		self.microbatch = microbatch if microbatch > 0 else batch_size
		self.lr = lr
		self.ema_rate = (
			[ema_rate]
			if isinstance(ema_rate, float)
			else [float(x) for x in ema_rate.split(",")]
		)
		self.log_interval = log_interval
		self.save_interval = save_interval
		self.resume_checkpoint = resume_checkpoint
		self.use_fp16 = use_fp16
		self.fp16_scale_growth = fp16_scale_growth
		self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
		self.weight_decay = weight_decay
		self.lr_anneal_steps = lr_anneal_steps
		self.main_data_indentifier = main_data_indentifier
		self.conditioning_variable = conditioning_variable
		self.cond_dropout_rate = cond_dropout_rate
		self.iterations = iterations

		# self.writer = SummaryWriter(logger.get_current() / 'tensorboard')
		self.step = 0
		self.resume_step = 0
		self.global_batch = self.batch_size * dist.get_world_size()

		self.sync_cuda = th.cuda.is_available()

		self._load_and_sync_parameters()
		self.mp_trainer = MixedPrecisionTrainer(
			model=self.model,
			use_fp16=self.use_fp16,
			fp16_scale_growth=fp16_scale_growth,
		)

		self.opt = AdamW(
			self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
		)
		if self.resume_step:
			self._load_optimizer_state()
			# Model was resumed, either due to a restart or a checkpoint
			# being specified at the command line.
			self.ema_params = [
				self._load_ema_parameters(rate) for rate in self.ema_rate
			]
		else:
			self.ema_params = [
				copy.deepcopy(self.mp_trainer.master_params)
				for _ in range(len(self.ema_rate))
			]

		if th.cuda.is_available():
			self.use_ddp = True
			self.ddp_model = DDP(
				self.model,
				device_ids=[dist_util.dev()],
				output_device=dist_util.dev(),
				broadcast_buffers=False,
				bucket_cap_mb=128,
				find_unused_parameters=True,
			)
		else:
			if dist.get_world_size() > 1:
				logger.warn(
					"Distributed training requires CUDA. "
					"Gradients will not be synchronized properly!"
				)
			self.use_ddp = False
			self.ddp_model = self.model

	def _load_and_sync_parameters(self):
		resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

		if resume_checkpoint:
			self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
			if dist.get_rank() == 0:
				logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
				self.model.load_state_dict(
					dist_util.load_state_dict(
						resume_checkpoint, map_location=dist_util.dev()
					)
				)

		dist_util.sync_params(self.model.parameters())

	def _load_ema_parameters(self, rate):
		ema_params = copy.deepcopy(self.mp_trainer.master_params)

		main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
		ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
		if ema_checkpoint:
			if dist.get_rank() == 0:
				logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
				state_dict = dist_util.load_state_dict(
					ema_checkpoint, map_location=dist_util.dev()
				)
				ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

		dist_util.sync_params(ema_params)
		return ema_params

	def _load_optimizer_state(self):
		main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
		opt_checkpoint = bf.join(
			bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
		)
		if bf.exists(opt_checkpoint):
			logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
			state_dict = dist_util.load_state_dict(
				opt_checkpoint, map_location=dist_util.dev()
			)
			self.opt.load_state_dict(state_dict)

	def run_loop(self):
		# return 0
		# run_loop是核心代码，首先获得数据集里面的一个批次数据batch以及对应条件cond。
		# 然后用run_step进行一次训练
		while (
				# not self.lr_anneal_steps    # =0
				# or self.step + self.resume_step < self.iterations  # iterations: int = 70e3,
				self.step < self.iterations
		):
			data_dict = next(self.data)  # data=train_loader, data_val=val_loader,
			self.run_step(data_dict)
			if self.step % self.save_interval == 0:  # training_score.save_interval = 10000
				self.save()
			if self.step % self.log_interval == 0:  # log_interval = 50
				for val_step in range(self.log_interval):
					val_data_dict = next(self.data_val)
					self.forward_backward(val_data_dict, phase="val")
				print('^^^^^^^^^^^^^')
				logger.dumpkvs()
				print('^^^^^^^^^^^^^')
			self.step += 1
		# Save the last checkpoint if it wasn't already saved.
		if (self.step - 1) % self.save_interval != 0:
			self.save()

	def run_step(self, data_dict):  # data=train_loader, data_val=val_loader,
		self.forward_backward(data_dict, phase="train")  # 前向传播与反向传播
		# forward添加噪声并记录损失函数，backward根据损失函数得出优化模型的梯度
		took_step = self.mp_trainer.optimize(self.opt)
		if took_step:
			self._update_ema()
		self._anneal_lr()
		self.log_step()

	def conditioning_dropout(self, model_conditionals: Dict):
		'''
        By setting the self.conditioning_variable to self.num_classes,
            we are telling the Embedding layer in the model to use non-class informative embedding (padding idx default to 0).
        '''

		idx2drop = int(model_conditionals["y"].shape[0] * self.cond_dropout_rate)
		# model_conditionals["y"][th.randint(model_conditionals["y"].shape[0],(idx2drop,))] = self.model.num_classes
		model_conditionals["y"][th.randint(model_conditionals["y"].shape[0], (idx2drop,))] = 2

		return model_conditionals

	def forward_backward(self, data_dict, phase: str = "train"):

		# self.main_data_indentifier = "image"
		# data_dictPosixPath('/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_val/patient_BraTS2021_00025'),
		batch = data_dict.pop(self.main_data_indentifier)
		# print('210 def forward_backward type(batch):{}'.format(type(batch)))     # <class 'torch.Tensor'>
		# print('210 value(batch):{}'.format(batch))     # <class 'torch.Tensor'>
		model_conditionals = data_dict
		# print('model_conditionals:')
		# print(model_conditionals)     <class 'dict'>
		# 'gt': (32, 4, 64, 64)
		# 'patient_id': tensor([1125, 838, 1205, 103, 1294, 210, 1213, 379, 1198, 459,
		#                       404, 1463, 16, 24, 1119, 1147, 1349, 1193, 1560, 242, 646,
		#                       1290, 604, 387, 149, 1578, 386, 1213, 1131, 651, 294, 1134]),
		# 'slice_id': tensor([ 45, 10, 27, 107, 98, 25, 103, 47, 65, 108, 115, 135, 79, 52, 79, 86,
		#                      73, 51, 101, 8, 98, 111, 55, 60, 9, 99, 74, 45, 108,  81, 125, 85]),
		# 'y': tensor([1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
		#              0, 1, 1, 0, 1, 0, 1, 1]),
		# 'conditioning_x':  (32, 4, 64, 64)
		assert phase in ["train", "val"]
		if phase == "train":  # model.train()是保证BN层用每一批数据的均值和方差
			self.ddp_model.train()
			if self.cond_dropout_rate != 0:  # =0.35
				# print('224 def forward_backward value(cond_dropout_rate):{}'.format(self.cond_dropout_rate))
				model_conditionals = self.conditioning_dropout(model_conditionals)
			# print('235 value of cond_dropout_rate:{}'.format(self.cond_dropout_rate))
		else:
			# print('valvalvalval')
			self.ddp_model.eval()  # model.eval()是保证BN用全部训练数据的均值和方差
			# print('valvalvalval finished')

		# 清空梯度
		self.mp_trainer.zero_grad()
		# 清空模型参数梯度，将模型参数的梯度缓存设置为0。
		# 在进行反向传播时，梯度会累加，如果不清空梯度，会影响后续的梯度计算

		# 对输入数据进行微批次处理
		for i in range(0, batch.shape[0], self.microbatch):
			# 获取当前微批次数据
			micro = batch[i: i + self.microbatch].to(dist_util.dev())
			# 获取当前微批次对应的条件
			micro_cond = {
				k: v[i: i + self.microbatch].to(dist_util.dev())
				for k, v in model_conditionals.items()
			}
			# 判断是否为最后一批此数据
			last_batch = (i + self.microbatch) >= batch.shape[0]
			# 从扰动采样其中采样噪声和对应的权重
			t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

			# 定义计算loss的函数
			compute_losses = functools.partial(
				self.diffusion.training_losses,
				self.ddp_model,
				micro,
				t,
				model_kwargs=micro_cond
			)

			# 如果使用DDP分布式训练，则使用no_sync()避免上下文同步
			if last_batch or not self.use_ddp:
				losses = compute_losses()
			else:
				with self.ddp_model.no_sync():
					losses = compute_losses()

			# 如果使用LossAwareSampler，则更新采样器的本地loss值
			if isinstance(self.schedule_sampler, LossAwareSampler):
				self.schedule_sampler.update_with_local_losses(
					t, losses["loss"].detach()
				)

			# 计算加权平均loss
			loss = (losses["loss"] * weights).mean()
			# 记录loss和loss的分量信息
			log_loss_dict(
				self.diffusion, t, {phase + '_' + k: v * weights for k, v in losses.items()}
			)

			# 反向传播
			if phase == "train":
				self.mp_trainer.backward(loss)

	def _update_ema(self):
		for rate, params in zip(self.ema_rate, self.ema_params):
			update_ema(params, self.mp_trainer.master_params, rate=rate)

	def _anneal_lr(self):
		if not self.lr_anneal_steps:
			return
		frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
		lr = self.lr * (1 - frac_done)
		for param_group in self.opt.param_groups:
			param_group["lr"] = lr

	def log_step(self):
		logger.logkv("step", self.step + self.resume_step)
		logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

	def save(self):
		def save_checkpoint(rate, params):
			state_dict = self.mp_trainer.master_params_to_state_dict(params)
			if dist.get_rank() == 0:
				logger.log(f"saving model {rate}...")
				if not rate:
					filename = f"model{(self.step + self.resume_step):06d}.pt"
				else:
					filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
				with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
					th.save(state_dict, f)

		save_checkpoint(0, self.mp_trainer.master_params)
		for rate, params in zip(self.ema_rate, self.ema_params):
			save_checkpoint(rate, params)

		if dist.get_rank() == 0:
			with bf.BlobFile(
					bf.join(get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"),
					"wb",
			) as f:
				th.save(self.opt.state_dict(), f)

		dist.barrier()


def parse_resume_step_from_filename(filename):
	"""
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
	split = filename.split("model")
	if len(split) < 2:
		return 0
	split1 = split[-1].split(".")[0]
	try:
		return int(split1)
	except ValueError:
		return 0


def get_blob_logdir():
	# You can change this to be a separate path to save checkpoints to
	# a blobstore or some external drive.
	return logger.get_dir()


def find_resume_checkpoint():
	# On your infrastructure, you may want to override this to automatically
	# discover the latest checkpoint on your blob storage, etc.
	return None


def matplotlib_imshow(img: th.tensor):
	img = torchvision.utils.make_grid(img.detach().cpu())
	img = img / 2 + 0.5  # unnormalize
	npimg = img.numpy()
	plt.imshow(npimg, cmap="Greys")


def find_ema_checkpoint(main_checkpoint, step, rate):
	if main_checkpoint is None:
		return None
	filename = f"ema_{rate}_{(step):06d}.pt"
	path = bf.join(bf.dirname(main_checkpoint), filename)
	if bf.exists(path):
		return path
	return None


def log_loss_dict(diffusion, ts, losses):
	for key, values in losses.items():
		values_mean = values.mean()
		values_cpu = values_mean.to('cpu')
		values_item = values_cpu.item()
		logger.logkv_mean(key, values_item)
		# Log the quantiles (four quartiles, in particular).
		'''for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)'''


def get_random_vector_excluding(vector_to_exclude: th.tensor, nb_classes: int = 10):
	def get_random_number_excluding(exclude):
		randInt = randint(0, nb_classes)
		return get_random_number_excluding(exclude) if randInt == exclude else randInt

	new_random_vector = th.tensor([get_random_number_excluding(number) for number in vector_to_exclude]).to()
	new_random_vector.to(vector_to_exclude.device)

	assert (new_random_vector != vector_to_exclude).all().numpy(), "tensors should be different"
	assert (new_random_vector.size() == vector_to_exclude.size()), "vectors should have the same size"

	return new_random_vector
