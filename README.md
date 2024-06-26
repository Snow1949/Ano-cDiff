
Ano-cDiff
===================================
This repository is the work of "Counterfactual condition diffusion with continuous prior adaptive correction for anomaly detection in multimodal brain MRI" based on pytorch implementation, currently under review by Journal Expert Systems with Applications. More details can be seen in our upcoming paper.

After the paper is accepted, we will continue to improve the content of the document, including the ideas of the paper and specific experimental methods.
![image](framework.png)

Requirements：
-----------------------------------
	python 		3.6
	pytorch 	1.8.1 or later CUDA version
	torchvision 0.8.1+ cu110
	nibabel		3.2.1
	SimpleITK	2.1.1.2
	matplotlib	3.3.3
	Pillow		8.0.1
File structure：
-----------------------------------
* configs/get_config.py - gets the parameter configurations from a particular data class
* configs/brats_configs.py - default configs of brats datasets
* datasets/data_preprocessing.py - preprocess '.nii.gz' data into '.npy'
* datasets/load_brats.py - custom dataset loader. 
* models/gaussian_diffusion.py - Gaussian architecture with custom detection, forked from [Ho et al's diffusion models](https://github.com/hojonathanho/diffusion/tree/1e0dceb3b3495bbe19116a5e1b3596cd0706c543)<br />
* models/unet.py - baseline architecture of denoising network
* sampling/sampling_utils.py - counterfactual estimation with implicit condition guidance
* sampling/Ano-cDiff_sample.py - counterfactual inference sampling
* training/Ano-cDiff_train.py - trains the diffusion model for learning counterfactual inference

Run for user:
-----------------------------------
### Dataset preprocessing,
First, run `MRI_preprocessing.py` to process `.nii.gz` data to `.npy`. 
The BrainDataset class then assigns image-level labels (label=0 or 1) to each slice.
	
	run 'MRI_preprocessing.py' like this:
	cd /chenxue/paper3/Ano-cDiff/baseline_code/datasets && python -u MRI_preprocessing.py
        
### Train: 
See default configuration setting in `brats_configs.py`
Sample Run Script:

	cd /chenxue/paper3/Ano-cDiff_baseline && python -u baseline_code/training/Ano-cDiff_train.py

### Sample:
See default configuration setting in `brats_configs.py`
Sample Run Script:

	cd /chenxue/paper3/Ano-cDiff_baseline && python -u baseline_code/sampling/Ano-cDiff_sample.py
 
### run for usr-provided data:
For the user-supplied data, the key to running the program is the construction of the BrainDataset class. First, all data is stored as shown in the `file_example.png`.

Citation
-----------------------------------
If you find this repo useful for your research, please consider citing the paper as follows:
	
	@article{chen2024counterfactual,
	  title={Counterfactual condition diffusion with continuous prior adaptive correction for anomaly detection in multimodal brain MRI},
	  author={Chen, Xue and Peng, Yanjun},
	  journal={Expert Systems with Applications},
	  pages={124295},
	  year={2024},
	  publisher={Elsevier}
	}
 
