"""
Like score_sampling.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import os
import numpy as np
import torch as th
import torch.distributed as dist
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import argparse
import random

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

from baseline_code.datasets import loader
from baseline_code.configs import get_config
from baseline_code.utils import logger, dist_util, script_util
from baseline_code.sampling.sampling_utils import get_models_functions, estimate_counterfactual

def main(args):
	config = get_config.file_from_dataset(args.dataset)
	print('+++++++config++++++++++')
	print(config)
	print('+++++++++++++++++++++++')

	dist_util.setup_dist()
	logger.configure(Path(config.experiment_name) / ("counterfactual_sampling_" + "_".join(config.classifier.label)))

	logger.log("creating loader...")
	test_loader = loader.get_data_loader(args.dataset, config, split_set='test', generator = False)

	logger.log("creating model and diffusion...")

	classifier, diffusion, model = script_util.get_models_from_config(config)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	logger.log(f"Number of parameteres: {pytorch_total_params}")    # 27147784

	cond_fn, model_fn, model_classifier_free_fn, denoised_fn = get_models_functions(config, model, classifier)

	model = th.nn.DataParallel(model, device_ids=[0, 1]).cuda()

	logger.log("sampling...")

	all_results = []
	all_images = []
	all_labels = []
	# test dataset lenght: 34847 (250个病人，共34847张切片)
	for i, data_dict in enumerate(test_loader):
		counterfactual_image, sampling_progression = estimate_counterfactual(config,
												diffusion, cond_fn, model_fn,
												model_classifier_free_fn, denoised_fn,
												data_dict)

		results_per_sample = {"original": data_dict,
							  "counterfactual_sample" : counterfactual_image.cpu().numpy(),
																}

		if config.sampling.progress:
			results_per_sample.update({"diffusion_process": sampling_progression})
			# print(len(sampling_progression))             # <class 'list'> 150
			# print(results_per_sample['sample'] - results_per_sample['pred_xstart'])     # sample pred_xstart
			print('---diffusion_process 233--')


		# config.device = 'cuda'
		# minbatch = diffusion.calc_bpd_loop(model, sampling_progression['pred_xstart'], clip_denoised=True, model_kwargs=config)

		all_results.append(results_per_sample)
	#
		if config.sampling.num_samples is not None and ((i+1) * config.sampling.batch_size) >= config.sampling.num_samples:
			break
	#
	all_results = {k: [dic[k] for dic in all_results] for k in all_results[0]}

	if dist.get_rank() == 0:
		out_path = os.path.join(logger.get_dir(), f"samples.npz")
		logger.log(f"saving to {out_path}")
		np.savez(out_path, all_results)

	dist.barrier()
	logger.log("sampling complete")


def reseed_random(seed):
	random.seed(seed)     # python random generator
	np.random.seed(seed)  # numpy random generator
	th.manual_seed(seed)
	th.cuda.manual_seed_all(seed)
	th.backends.cudnn.deterministic = True
	th.backends.cudnn.benchmark = False

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="mnist or brats", type=str, default='brats')
	args = parser.parse_args()
	print(args.dataset)
	main(args)