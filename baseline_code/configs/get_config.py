# import brats_configs
from baseline_code.configs import brats_configs

def file_from_dataset(dataset_name):
    if dataset_name == "brats":
        return brats_configs.get_default_configs()
    # elif dataset_name == "zhuanyiliu":
    #     return zhuanyiliu_configs.get_default_configs()
    else:
        raise Exception("Dataset not defined.")
