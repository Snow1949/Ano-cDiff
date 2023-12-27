#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
import torch
import random
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch.nn.functional as F


def normalise_percentile(volume):
	"""
    Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.
    """
	for mdl in range(volume.shape[1]):
		v_ = volume[:, mdl, :, :].reshape(-1)
		v_ = v_[v_ > 0]  # Use only the brain foreground to calculate the quantile
		p_99 = torch.quantile(v_, 0.99)
		volume[:, mdl, :, :] /= p_99

	return volume


def process_patient(path, target_path):
	flair = nib.load(path / f"{path.name}_flair.nii.gz").get_fdata()
	t1 = nib.load(path / f"{path.name}_t1.nii.gz").get_fdata()
	t1ce = nib.load(path / f"{path.name}_t1ce.nii.gz").get_fdata()
	t2 = nib.load(path / f"{path.name}_t2.nii.gz").get_fdata()
	labels = nib.load(path / f"{path.name}_seg.nii.gz").get_fdata()

	volume = torch.stack([torch.from_numpy(x) for x in [flair, t1, t1ce, t2]], dim=0).unsqueeze(dim=0)
	labels = torch.from_numpy(labels > 0.5).float().unsqueeze(dim=0).unsqueeze(dim=0)

	patient_dir = target_path / f"patient_{path.name}"
	patient_dir.mkdir(parents=True, exist_ok=True)

	volume = normalise_percentile(volume)

	sum_dim2 = (volume[0].mean(dim=0).sum(axis=0).sum(axis=0) > 0.5).int()
	fs_dim2 = sum_dim2.argmax()
	ls_dim2 = volume[0].mean(dim=0).shape[2] - sum_dim2.flip(dims=[0]).argmax()

	for slice_idx in range(fs_dim2, ls_dim2):
		low_res_x = F.interpolate(volume[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))
		low_res_y = F.interpolate(labels[:, :, :, :, slice_idx], mode="bilinear", size=(128, 128))

		np.savez_compressed(patient_dir / f"slice_{slice_idx}", x=low_res_x, y=low_res_y)


def preprocess(datapath: Path):
	# datapath = Path("chenxue/dataset/brain/MICCAI_BraTS2020/")

	all_imgs = sorted(list((datapath).iterdir()))
	print("all_imgs:")
	print(all_imgs)

	# splits_path = Path(__file__).parent.parent / "data" / "brats2020_preprocessed" / "data_splits"
	splits_path = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/")
	print('splits_path:')
	print(splits_path)

	path_train = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/TrainingData/")
	path_val = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/ValidationData/")
	path_test = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/TestingData/")
	all_imgs_train = sorted(list((path_train).iterdir()))
	all_imgs_val = sorted(list((path_val).iterdir()))
	all_imgs_test = sorted(list((path_test).iterdir()))

	indices_train = list(range(len(all_imgs_train)))
	indices_val = list(range(len(all_imgs_val)))
	indices_test = list(range(len(all_imgs_test)))
	random.seed(0)
	random.shuffle(indices_train)
	random.shuffle(indices_val)
	random.shuffle(indices_test)

	n_train = int(len(indices_train))
	n_val = int(len(indices_val))
	n_test = int(len(indices_test))
	print("n_train:{}".format(n_train))
	print("n_val:{}".format(n_val))
	print("n_test:{}".format(n_test))

	split_indices = {}
	split_indices["train"] = indices_train[:n_train]
	split_indices["val"] = indices_val[:n_val]
	split_indices["test"] = indices_test[:n_test]

	print('*********')
	print(split_indices["train"])
	print('*********')

	(splits_path / "train").mkdir(parents=True, exist_ok=True)
	with open(splits_path / "train" / "scans.csv", "w") as f:
		f.write("\n".join([all_imgs_train[idx].name for idx in split_indices["train"]]))

	(splits_path / "val").mkdir(parents=True, exist_ok=True)
	with open(splits_path / "val" / "scans.csv", "w") as f:
		f.write("\n".join([all_imgs_val[idx].name for idx in split_indices["val"]]))

	(splits_path / "test").mkdir(parents=True, exist_ok=True)
	with open(splits_path / "test" / "scans.csv", "w") as f:
		f.write("\n".join([all_imgs_test[idx].name for idx in split_indices["test"]]))

	# for split in ["train", "val", "test"]:
	# 	(splits_path / split).mkdir(parents=True, exist_ok=True)
	# 	with open(splits_path / split / "scans.csv", "w") as f:
	# 		f.write("\n".join([all_imgs[idx].name for idx in split_indices[split]]))
	#
	# for split in ["train", "val", "test"]:
	paths = [path_train / x.strip() for x in open(splits_path / "train" / "scans.csv").readlines()]
	print(f"Patients in train]: {len(paths)}")
	for source_path in tqdm(paths):
		# target_path = Path(__file__).parent.parent / "data" / "brats2020_preprocessed" / f"npy_{split}"
		target_path = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/") / f"npy_train"
		print("source_path:{}".format(source_path))
		print("target_path:{}".format(target_path))
		process_patient(source_path, target_path)

	paths = [path_val / x.strip() for x in open(splits_path / "val" / "scans.csv").readlines()]
	print(f"Patients in val]: {len(paths)}")
	for source_path in tqdm(paths):
		# target_path = Path(__file__).parent.parent / "data" / "brats2020_preprocessed" / f"npy_{split}"
		target_path = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/") / f"npy_val"
		print("source_path:{}".format(source_path))
		print("target_path:{}".format(target_path))
		process_patient(source_path, target_path)

	paths = [path_test / x.strip() for x in open(splits_path / "test" / "scans.csv").readlines()]
	print(f"Patients in test]: {len(paths)}")
	for source_path in tqdm(paths):
		# target_path = Path(__file__).parent.parent / "data" / "brats2020_preprocessed" / f"npy_{split}"
		target_path = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/") / f"npy_test"
		print("source_path:{}".format(source_path))
		print("target_path:{}".format(target_path))
		process_patient(source_path, target_path)

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", help="mnist or brats", type=str, default='brats')
	args = parser.parse_args()
	print(args.dataset)

	# datapath = Path(args.source)
	datapath = Path("/chenxue/dataset/brain/MICCAI_BraTS2020/")

	preprocess(datapath)


# cd /chenxue/code-2305/AnoDDPM2-master && python -u diffusion_training.py 28
# cd /chenxue/paper3/Ano-cDiff/baseline_UAD/datasets && python -u data_preprocessing.py
# cd /chenxue/paper3/Ano-cDiff/baseline_code/datasets && python -u data_preprocessing.py
# cd /chenxue/paper3/Ano-cDiff/baseline_HPEB/datasets && python -u data_preprocessing.py
# cd /chenxue/paper3/Ano-cDiff/baseline_HPEB_RFEN/datasets && python -u data_preprocessing.py
# cd /chenxue/paper3/Ano-cDiff/baseline_HPEB_RFEN_APAM/datasets && python -u data_preprocessing.py
# cd /chenxue/paper3/Ano-cDiff/optimal_model/datasets && python -u data_preprocessing.py