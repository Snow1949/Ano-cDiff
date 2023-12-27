import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset

i = 0


class PatientDataset(torch.utils.data.Dataset):
	def __init__(self, patient_dir: Path, process_fun=None, id=None, skip_condition=None, cache=True):
		"""
        Dataset class to store one patient's slices saved using np.savez_compressed.
        类来存储使用np.savez_compressed保存的一个病人的切片

        :param patient_dir Path to a dir containing saved slice .npz files.
        :param process_fun Processing on the fly function that takes the saved items as parameters
        :param id Patient id if available. For convenience and keeping track of patients.
        :param skip_condition predicate function to determine which slices should be excluded from the dataset.
         Takes the item after processing as an input.
        :param cache Whether to save/use cached lists of filtered indices to save time and avoid having to filter slices
         on object init.
        """

		self.patient_dir = patient_dir
		self.slice_paths = sorted(list(patient_dir.iterdir()))
		self.process = process_fun
		self.skip_condition = skip_condition
		self.id = id
		self.len = len(self.slice_paths)  # len(slice-"patient_BraTS2021_00723")=135
		self.idx_map = {x: x for x in range(self.len)}

		# print("self.len init:{}".format(int(self.len)))
		if self.skip_condition is not None:  # Perform some filtering based on 'skip_condition' predicate.
			import hashlib
			# try to generate a name for caching that depends on the patient_dir and skip_condition so that old cache is
			# not used when one of those is changed.
			# hashing of a function is tricky so this shouldn't be relied on too much...
			hash_object = hashlib.sha256(
				(str(patient_dir)).encode("utf-8") + str(skip_condition.__code__.co_code).encode("utf-8"))
			name = hash_object.hexdigest()
			if (self.patient_dir.parent.parent / "valid_indices_cache" / f"{name}.pkl").exists() and cache:
				import pickle
				self.idx_map = pickle.load(
					open((self.patient_dir.parent.parent / "valid_indices_cache" / f"{name}.pkl"), "rb"))
				self.len = len(self.idx_map)
				# print("self.len in exists():{}".format(int(self.len)))
			else:
				# Try and find which slices should be skipped and thus determine the length of the dataset.
				# 尝试查找应该跳过哪些片，从而确定数据集的长度。
				valid_indices = []
				for idx in range(self.len):  # self.len = 135
					# print("idx:{}".format(idx))
					global i
					i = i + 1
					with np.load(self.slice_paths[idx]) as data:
						if self.process is not None:
							item = self.process(**data)
						else:
							item = data
						if not skip_condition(item):
							valid_indices.append(idx)
							# print('slice_paths:{}, idx:{}'.format(self.slice_paths[idx], idx))
				self.len = len(valid_indices)
				# print("self.len in skip:{}".format(int(self.len)))
				self.idx_map = {x: valid_indices[x] for x in range(self.len)}
				# print("self.idx_map:{}".format(self.idx_map))
				if cache:
					import pickle
					cache_dir = self.patient_dir.parent.parent / "valid_indices_cache"
					cache_dir.mkdir(exist_ok=True)
					pickle.dump(self.idx_map, open(cache_dir / f"{name}.pkl", "wb"))
			print("self.idx_map:{}".format(self.idx_map))

	def __getitem__(self, idx):
		idx = self.idx_map[idx]  # 遍历slice(例：patient_BraTS2021_00723)文件夹下每一个slice
		path = str(self.slice_paths[idx])  # patient_BraTS2021_00723
		# print("idx:{}".format(idx))       # 例：80
		# print("path:{}".format(path))
		# path:/chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_train/patient_BraTS2021_01325/slice_46.npz
		data = np.load(path)
		patient_str = "patient_BraTS2021_"
		slice_str = "slice_"
		patient_id = int(path[path.find(patient_str) + len(patient_str):path.rfind("/")])
		slice_id = int(path[path.find(slice_str) + len(slice_str):path.rfind(".npz")])
		# print("patient_id:{}".format(patient_id))
		# print("slice_id:{}".format(slice_id))
		if self.process is not None:
			item = self.process(**data)
		else:
			item = (data,)
		# for name in item:
		# 	print('item[i]:{}'.format(name.shape))
		# print("item".format(item['x'].shape))
		# print("item[x].shape".format(item['x'].shape))
		# print("item[y].shape".format(item['y'].shape))
		return item + (patient_id, slice_id)

	def __len__(self):
		return self.len


class BrainDataset(torch.utils.data.Dataset):

	def __init__(self, datapath: Path, dataset="brats2021_64x64", split="val",
	             n_tumour_patients=None, n_healthy_patients=None,
	             scale_factor=1, binary=True, pad=None,
	             skip_healthy_s_in_tumour=False,
	             skip_tumour_s_in_healthy=True,
	             seed=0, cache=True, use_channels: List[int] = None, sequence_translation: bool = False,
	             norm_around_zero: bool = True):
		"""
        Dataset class for training/evaluation with the option to have datasets for semi-supervision.
        用于训练/评估的数据集类，具有用于半监督的数据集选项
        :param dataset: dataset identifier
        :param split: "train", "val" or "test".
        :param n_tumour_patients: number of patients w/ tumours to use. All slices (including slices containing tumours)
         from these patients will be included in the dataset
        :param n_healthy_patients: number of patients w/o tumours to use. Only slices not containing any tumour gt will
         be included from these patients.
        :param scale_factor: For resizing data on the fly.用于动态调整数据大小。
        :param binary: Whether to provide binary ground truth (background vs whole tumour) or possibly more granular
         classes (e.g. available in BraTS datasets).是否提供二元ground truth(背景与整个肿瘤)或可能更细粒度的分类(例如在BraTS数据集中可用)。
        :param pad: Padding for data/gt.
        :param skip_healthy_s_in_tumour: Whether to skip healthy slices in "tumour" patients
         (e.g. for testing/visualising results).
        :param skip_tumour_s_in_healthy: whether to skip tumour slices in healthy patients. Usually yes, unless for debugging, etc.
        :param seed:
        :param cache: Whether to use caching for filtering slices是否使用缓存过滤切片
        :param use_channels: Whether to use a subset of modalities. E.g. [0, 1] for FLAIR and T1 in BraTS.
        是否使用模态的子集。例如，BraTS中的FLAIR和T1为[0,1]。
        """
		# datapath: path："chenxue/dataset/brain/MICCAI_BraTS2020"
		self.rng = random.Random(seed)
		self.sequence_translation = sequence_translation
		self.use_channels = use_channels if use_channels is not None else list(range(4))  # 是否用模态子集
		self.split = split  # "train", "val" or "test"
		train_path = datapath / "npy_train"
		# /chenxue/Diff-SCM-main/diff_scm/datasets/data/brats2021_preprocessed/npy_train
		val_path = datapath / "npy_val"
		test_path = datapath / "npy_test"

		if split == "train":
			path = train_path
		elif split == "val":
			path = val_path
		elif split == "test":
			path = test_path
		else:
			raise ValueError(f"split {split} unknown")

		# Assuming item[1] is the gt
		if binary:
			self.skip_tumour = lambda item: item[1].sum() > 0
			self.skip_healthy = lambda item: item[1].sum() < 1
		else:
			self.skip_tumour = lambda item: item[1][1:, ...].sum() > 5
			self.skip_healthy = lambda item: item[1][1:, ...].sum() < 5

		# print("self.skip_tumour:{}".format(self.skip_tumour))
		# print("type:{}".format(type(self.skip_tumour)))                     #  type:<class 'function'>
		# print("self.skip_healthy:{}".format(self.skip_healthy))

		# On the fly preprocessing function 动态预处理功能
		def process(x, y, coords=None):
			if binary:
				# treat all tumour classes (or just WM lesions) as one for anomaly detection purposes.
				# 将所有肿瘤类别(或仅WM病变)视为一个以进行异常检测
				y = y > 0.5
			else:
				# convert to one-hot gt encoding 转换为one-hot gt编码
				y = np.concatenate([y == x for x in range(4)], axis=1)

			if scale_factor != 1:  # Rescale on the fly if needed如果需要，可以动态地重新缩放
				x = F.interpolate(torch.from_numpy(x).float(), scale_factor=scale_factor, mode="bilinear",
				                  align_corners=False, recompute_scale_factor=True)
				y = F.interpolate(torch.from_numpy(y).float(), scale_factor=scale_factor, mode="bilinear",
				                  align_corners=False, recompute_scale_factor=True)
				if coords is not None:
					coords = F.interpolate(torch.from_numpy(coords), scale_factor=scale_factor, mode="bilinear",
					                       align_corners=False, recompute_scale_factor=True)

			if pad is not None:
				x = F.pad(x, pad=pad)
				y = F.pad(y, pad=pad)
				if coords is not None:
					coords = F.pad(coords, pad=pad)

			if norm_around_zero:
				x = x * 2 - 1

			# 将默认pytorch dataloader中的默认collate转换为适当形状的torch张量。
			# Convert to torch tensors of appropriate shape for the default collate in default pytorch dataloader.
			# if coords is not None:
			#     return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float(), torch.from_numpy(coords[0]).float()
			# else:
			#     return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float(), None
			return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float()

		patient_dirs = sorted(list(path.iterdir()))
		print('*************')
		print("test patient_dirs:{}".format(len(patient_dirs)))      # patient_dirs:939
		print(patient_dirs)
		print('*************')
		if split == "train":
			self.rng.shuffle(patient_dirs)

		# assert ((n_tumour_patients is not None) or (n_healthy_patients is not None))
		self.n_tumour_patients = n_tumour_patients if n_tumour_patients is not None else len(patient_dirs)
		self.n_healthy_patients = n_healthy_patients if n_healthy_patients is not None else len(
			patient_dirs) - self.n_tumour_patients
		# self.n_tumour_patients:939
		# self.n_healthy_patients:0
		print("self.n_tumour_patients:{}".format(self.n_tumour_patients))  # self.n_tumour_patients:939
		print("self.n_healthy_patients:{}".format(self.n_healthy_patients))  # self.n_healthy_patients:0

		# 肿瘤患者(例如，在半监督的情况下，提供一些肿瘤标签进行培训)
		# Patients with tumours (e.g. for a semi-supervised case where some tumour labels are provided for training)
		self.patient_datasets = [PatientDataset(patient_dirs[i], process_fun=process, id=i, cache=cache,
		                                        skip_condition=self.skip_healthy if skip_healthy_s_in_tumour else None)
		                         for i in range(self.n_tumour_patients)]
		print("patient with tumours finished")
		# + only healthy slices from "healthy" patients; +only“健康”病人的健康切片
		self.patient_datasets += [PatientDataset(patient_dirs[i],
		                                         skip_condition=self.skip_tumour if skip_tumour_s_in_healthy else None,
		                                         cache=cache, process_fun=process, id=i)
		                          for i in
		                          range(self.n_tumour_patients, self.n_tumour_patients + self.n_healthy_patients)]
		print("only healthy slices finished")
		self.dataset = ConcatDataset(self.patient_datasets)

	def __getitem__(self, idx):
		item = {}
        # print(self.dataset[idx])
        # for k, v in self.dataset[idx].ite:
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # item['image'], item['gt'], item['coords'], item["patient_id"], item["slice_id"] = self.dataset[idx]
		item['image'], item['gt'], item["patient_id"], item["slice_id"] = self.dataset[idx]
		print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
		print("split:{}, patient_id:{}, slice_id:{}".format(self.split, item["patient_id"], item["slice_id"]))
		print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
		if self.sequence_translation and (self.split != "test"):  
			sequence = self.rng.choice(self.use_channels)
			item['image'] = item['image'][sequence].unsqueeze(0)
			item['y'] = sequence
			# print("1111111111")
		else:
			# if at least one voxel belongs has a tumour mask, it's an unhealthy slice
			is_slice_healthy = torch.amax(item['gt'], dim=(0, 1, 2)).to(torch.long)
			item['y'] = is_slice_healthy
			# print("2222222")
		conditioning_x = item["image"].detach().clone()
		# passing brain mask for maintaining brain shape
		brain_mask = conditioning_x > -1
		# print("++++++")
		if np.random.uniform() > 0.5:
			conditioning_x[..., :conditioning_x.shape[-2] // 2, :] = -1
		else:
			conditioning_x[..., conditioning_x.shape[-2] // 2:, :] = -1
		# print("++++++")
		# torch.cat([conditioning_x,brain_mask[:1]],axis = 0)
		item["conditioning_x"] = brain_mask[:1].to(torch.float)
		return item

	def __len__(self):
		return len(self.dataset)


def re_write_file_names_to_include_healthy_status(path):
	paths = list(Path(path).glob("*/**.npz"))
