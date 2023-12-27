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

from diff_scm.datasets import loader
from diff_scm.configs import get_config
from diff_scm.utils import logger, dist_util, script_util
from diff_scm.sampling.sampling_utils import get_models_functions, estimate_counterfactual

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

    logger.log(f"Number of parameteres: {pytorch_total_params}")

    cond_fn, model_fn, model_classifier_free_fn, denoised_fn = get_models_functions(config, model, classifier)

    logger.log("sampling...")

    all_results = []
    all_images = []
    all_labels = []
    for i, data_dict in enumerate(test_loader):
        print('i:')
        print(str(i))
        counterfactual_image, sampling_progression = estimate_counterfactual(config, 
                                                diffusion, cond_fn, model_fn, 
                                                model_classifier_free_fn, denoised_fn, 
                                                data_dict)
            
        results_per_sample = {"original": data_dict,
                              "counterfactual_sample" : counterfactual_image.cpu().numpy(),
                                                                }

        # print("original type:")                             # <class 'dict'>
        # print(type(data_dict))
        # # print('original image', data_dict[0].shape, data_dict[1])
        # for key in data_dict:
        #     print(key)
        #     print(type(data_dict[key]))
        #     print(data_dict[key].shape)
        #     print(data_dict[key])
        # # image;    gt;    patient_id;  slice_id;   y;  conditioning_x;
        # # <class 'torch.Tensor'>        ...
        # # torch.Size([32, 4, 64, 64])   ([32, 1, 64, 64])    ([32])   ([32])   ([32])   ([32, 1, 64, 64])
        #

        # print('++++++++++0815 test++++++++++++')
        # print('sample input:')
        # input_data = ((data_dict['image'] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # input_gt = ((data_dict['gt'] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        # print('type(input_data):{}'.format(type(input_data)))
        # for ith in range(32):
        #     input_img0 = input_data[ith, 0, ...]
        #     print('type(input_img0):{}'.format(type(input_img0)))
        #     print('shape(input_img0):{}'.format(input_img0.shape))          # torch.Size([64, 64])
        #     input_img1 = input_data[ith, 1, ...]
        #     input_img2 = input_data[ith, 2, ...]
        #     input_img3 = input_data[ith, 3, ...]
        #
        #     # input_gt = input_gt[ith, 0, ...]
        #     print('type(gt):{}'.format(type(input_gt)))                           # torch.Size([64, 64])
        #     print('shape(gt):{}'.format(input_gt.shape))
        #     # images_tensor = th.cat([input_img0, input_img1, input_img2, input_img3, input_gt], dim=1)
        #     images_tensor = th.cat([input_img0, input_img1, input_img2, input_img3], dim=1)
        #     print('images.size:{}'.format(images_tensor.size()))       # torch.size([64, 320])
        #     images_numpy = images_tensor.cpu().numpy()
        #     images = Image.fromarray(np.uint8(images_numpy))
        #     # save_image(images, 'input_png')
        #     images.save('/chenxue/experiment_data/anomaly_diffusion_all_train_brats2021_64x64/input_img/input_' + str(ith) + '.jpg')
        #

        print('sample output:')             # (32, 4, 64, 64)
        print('type(counterfactual_image):{}'.format(type(counterfactual_image)))
        output_data = (counterfactual_image + 1) * 127.5
        # output_gt = ((counterfactual_image['gt'] + 1) * 127.5).clamp(0, 255).to(th.uint8)
        print('type(output_data):{}'.format(type(output_data)))
        for ith in range(32):
            output_img0 = output_data[ith, 0, ...]
            print('type(output_img0):{}'.format(type(output_img0)))
            print('shape(output_img0):{}'.format(output_img0.shape))  # torch.Size([64, 64])
            output_img1 = output_data[ith, 1, ...]
            output_img2 = output_data[ith, 2, ...]
            output_img3 = output_data[ith, 3, ...]

            # input_gt = input_gt[ith, 0, ...]
            # print('type(gt):{}'.format(type(input_gt)))                           # torch.Size([64, 64])
            # print('shape(gt):{}'.format(input_gt.shape))
            # images_tensor = th.cat([input_img0, input_img1, input_img2, input_img3, input_gt], dim=1)
            images_tensor_out = th.cat([output_img0, output_img1, output_img2, output_img3], dim=1)
            print('images.size:{}'.format(images_tensor_out.size()))  # torch.size([64, 320])
            images_numpy_out = images_tensor_out.cpu().numpy()
            images = Image.fromarray(np.uint8(images_numpy_out))
            # save_image(images, 'input_png')
            images.save('/chenxue/experiment_data/anomaly_diffusion_all_train_brats2021_64x64/output_img/output_' + str(ith) + '.jpg')


        print("counterfactual_sample type:")                # <class 'numpy.ndarray'>
        print(type(counterfactual_image.cpu().numpy()))
        print("counterfactual_sample shape:")
        print(counterfactual_image.cpu().numpy().shape)     # (32, 4, 64, 64)
        print(counterfactual_image.cpu().numpy().size)      # 524288
        if config.sampling.progress:
            results_per_sample.update({"diffusion_process": sampling_progression})
            print("diffusion_process type:")                # <class 'list'>
            print(type(sampling_progression))
                                                        
        all_results.append(results_per_sample)

        if config.sampling.num_samples is not None and ((i+1) * config.sampling.batch_size) >= config.sampling.num_samples:
            break                

    all_results = {k: [dic[k] for dic in all_results] for k in all_results[0]}

    print('type(all_results):{}'.format(type(all_results)))         # type(all_results):<class 'dict'>  len=150
    # for key in all_results:
    #     print(key)
    #     print(type(all_results[key]))
    #     print(len(all_results[key]))
    #     print(all_results[key])
    #   original    counterfactual_sample   diffusion_process
    #   <class 'list'>
    #   1
    diffusion_process = all_results['diffusion_process']    # <list>
    for i, element in enumerate(diffusion_process[0]):      # type(element) = <class 'dict'>
        print(i)
        for key in element:
            print(key)                  # sample  pred_xstart   score_mean
            print(type(element[key]))   # <class 'torch.Tensor'>
            print(element[key].shape)   # torch.Size([32, 4, 64, 64])
            # print(element[key])
            print('*************************************************')
            print('element[key]:')
            result_data = ((element[key] + 1) * 127.5).clamp(0, 255).to(th.uint8)
            print('type(result_data):{}'.format(type(result_data)))
            for ith in range(32):
                result_img0 = result_data[ith, 0, ...]
                print('type(input_img0):{}'.format(type(result_img0)))
                print('shape(input_img0):{}'.format(result_img0.shape))          # torch.Size([64, 64])
                result_img1 = result_data[ith, 1, ...]
                result_img2 = result_data[ith, 2, ...]
                result_img3 = result_data[ith, 3, ...]

                images_tensor = th.cat([result_img0, result_img1, result_img2, result_img3], dim=1)
                print('images.size:{}'.format(images_tensor.size()))       # torch.size([64, 320])
                images_numpy = images_tensor.cpu().numpy()
                images = Image.fromarray(np.uint8(images_numpy))
                # save_image(images, 'input_png')
                images.save('/chenxue/experiment_data/anomaly_diffusion_all_train_brats2021_64x64/result_img/' + str(key) + '_' + str(ith) + '.jpg')
            print('*************************************************')

        # print(all_results[key])


    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), f"samples.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, all_results)
        print('********samples results********')
        print(type(all_results))        # <class 'dict'>
        print(len(all_results))         # 3
        # print(all_results[key])
        print(type(all_results['original']))    # <class 'list'>
        print(len(all_results['original']))     # 1
        # print(all_results[key])
        print(type(all_results['counterfactual_sample']))   # <class 'list'>
        print(len(all_results['counterfactual_sample']))    # 1
        print(all_results['counterfactual_sample'])
        print(type(all_results['diffusion_process']))       # <class 'list'>
        print(len(all_results['diffusion_process']))        # 1
        print(all_results['diffusion_process'])

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