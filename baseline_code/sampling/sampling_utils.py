from typing import Dict
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import numpy as np
from baseline_code.utils import logger, dist_util
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import seaborn as sns


def estimate_counterfactual(config, diffusion, cond_fn, model_fn, model_classifier_free_fn, denoised_fn, data_dict):
    model_kwargs, init_image = get_input_data(config, data_dict)
    # DDIM loop in reverse time order for inferring exogenous noise (image latent space)
    # 反向时间顺序DDIM环路用于推断外源噪声(图像潜在空间),前向加噪
    exogenous_noise, abduction_progression = diffusion.ddim_sample_loop(
            model_fn,
            (config.sampling.batch_size,
                config.score_model.num_input_channels,
                config.score_model.image_size,
                config.score_model.image_size),
            clip_denoised=config.sampling.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn = denoised_fn if config.sampling.dynamic_sampling else None,
            noise=init_image,
            cond_fn=None,
            device=dist_util.dev(),
            progress=config.sampling.progress,
            eta=config.sampling.eta,
            reconstruction=True,
            sampling_progression_ratio = config.sampling.sampling_progression_ratio
        )
    init_image = exogenous_noise            # 前向扩散重点噪声图 <class 'torch.Tensor'>
    print('shape(exogenous_noise):{}'.format(init_image.shape))         # torch.Size([64, 4, 64, 64])
    init_image_data = ((init_image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    print('shape(input_image_data):{}'.format(init_image_data.shape))   # torch.Size([64, 4, 64, 64])
    for ith in range(64):
        input_img0 = init_image_data[ith, 0, ...]
        # img0_numpy = input_img0.cpu().numpy()
        # sns.set()
        # sns.heatmap(img2, xticklabels=False,yticklabels=False,cmap="coolwarm",cbar=False)
        # sns.heatmap(data=img0_numpy, cmap="coolwarm", vmax=25, vmin=0)
        # plt.title("test")
        # plt.savefig('exogenous_noise/heatmap_' + str(ith) + '.png')

        print('shape(input_img0):{}'.format(input_img0.shape))          # torch.Size([64, 64])
        input_img1 = init_image_data[ith, 1, ...]
        input_img2 = init_image_data[ith, 2, ...]
        input_img3 = init_image_data[ith, 3, ...]
        # input_gt = input_gt[ith, 0, ...]
        # print('type(gt):{}'.format(type(input_gt)))                           # torch.Size([64, 64])
        # print('shape(gt):{}'.format(input_gt.shape))
        # images_tensor = th.cat([input_img0, input_img1, input_img2, input_img3, input_gt], dim=1)
        images_tensor = torch.cat([input_img0, input_img1, input_img2, input_img3], dim=1)
        print('images.size:{}'.format(images_tensor.size()))       # torch.size([64, 256])
        images_numpy = images_tensor.cpu().numpy()
        images = Image.fromarray(np.uint8(images_numpy))
        images = images.convert(mode="RGB", colors='PuRd')
        images.save('exogenous_noise/latent_' + str(ith) + '.png')
    print("latent space sampling finished")

    # DDIM diffusion inference  with conditioning (intervention), starting from a latent image instead of random noise
    counterfactual_image, diffusion_progression = diffusion.ddim_sample_loop(
            model_classifier_free_fn if config.score_model.classifier_free_cond else model_fn,
            (config.sampling.batch_size,
                config.score_model.num_input_channels,
                config.score_model.image_size,
                config.score_model.image_size),
            clip_denoised=config.sampling.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn = denoised_fn if config.sampling.dynamic_sampling else None,
            noise=init_image,
            cond_fn=cond_fn if config.sampling.classifier_scale != 0 else None,
            device=dist_util.dev(),
            progress=config.sampling.progress,
            eta=config.sampling.eta,
            reconstruction=False,
            sampling_progression_ratio = config.sampling.sampling_progression_ratio
        )
    sampling_progression = abduction_progression + diffusion_progression

    latent_image = counterfactual_image  # 前向扩散重点噪声图 <class 'torch.Tensor'>
    print('shape(latent_image):{}'.format(latent_image.shape))  # torch.Size([64, 4, 64, 64])

    latent_image_data = ((latent_image + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    print('shape(input_data):{}'.format(latent_image_data.shape))   # torch.Size([64, 4, 64, 64])
    for ith in range(64):
        input_img0 = latent_image_data[ith, 0, ...]
        print('ith shpe:{}'.format(latent_image_data[ith, ...].shape))  # torch.Size([4, 64, 64])
        # img0_numpy = input_img0.cpu().numpy()
        # sns.set()
        # sns.heatmap(img2, xticklabels=False,yticklabels=False,cmap="coolwarm",cbar=False)
        # sns.heatmap(data=img0_numpy, cmap="coolwarm", vmax=25, vmin=0)
        # plt.title("test")
        # plt.savefig('exogenous_noise/heatmap_sample_' + str(ith) + '.png')

        print('shape(input_img0):{}'.format(input_img0.shape))  # torch.Size([64, 64])
        input_img1 = latent_image_data[ith, 1, ...]
        input_img2 = latent_image_data[ith, 2, ...]
        input_img3 = latent_image_data[ith, 3, ...]

        images_tensor = torch.cat([input_img0, input_img1, input_img2, input_img3], dim=1)
        print('generate images.size:{}'.format(images_tensor.size()))   # torch.Size([64, 256])
        images_numpy = images_tensor.cpu().numpy()

        images = Image.fromarray(np.uint8(images_numpy))
        print('###################################')

        # # 修改权重图颜色
        # datas = images.getdata()
        # print("type(datas):{}".format(type(datas)))
        # pixels = list(datas)
        # print("pixels:{}".format(pixels))
        # import statistics
        # common = statistics.mode(pixels)
        # print(type(common))
        # print('common:{}'.format(common))
        # print('+++++++print datas++++++++++')
        # print(datas)
        # print('+++++++++print datas finished ++++++')
        # for i, data in np.ndenumerate(datas):
        #     print('i:{}, data:{}'.format(i, data))
        # print('******')
        # # print(type(datas), datas[1]): <class 'ImagingCore'> (0, 0, 0, 0)
        # newData = []
        # import random
        # for item in datas:
        #     if item[0] < 10 and item[1] < 10 and item[2] < 10:
        #         # newData.append((226, 226, 240))
        #         newData.append((common[0] + random.randint(0, 3), common[1] + random.randint(0, 3),
        #                         common[2] + random.randint(0, 3)))
        #     else:
        #         newData.append(item)
        # images.putdata(newData)
        print('###################################')



        images = images.convert(mode="RGB", colors='PuRd')
        # save_image(images, 'input_png')
        images.save('exogenous_noise/generate_' + str(ith) + '.png')
    print('######print finsihed######')

    return counterfactual_image, sampling_progression

def get_models_functions(config, model, anti_causal_predictor):
    def cond_fn(x, t, y=None, **kwargs):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            out = anti_causal_predictor(x_in, t)
            # print('++++++++out+++++++++++++')
            # print('type(out):{}'.format(type(out)))
            # print(out)
            # print('++++++++out+++++++++++++')
            if isinstance(out, Dict):
                logits = out[config.sampling.label_of_intervention]     # out['gt']
                print('^^^^^^logits^^^^^^^^^^')
                print(logits)
                print('^^^^^^^^^^^^^^^^^^^^^^')
            else:
                logits = out
                print('------logits----------')
                print(logits)
                print('----------------------')


            ## deal with
            y_new = torch.cat(2*[y[:y.size()[0]//2]]) if y.max() >= logits.size()[-1] else y
                
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y_new.view(-1)]
            grad_log_conditional = torch.autograd.grad(selected.sum(), x_in)[0]

            return grad_log_conditional * config.sampling.classifier_scale  # * scaling[:, None, None, None]

    def model_fn(x, t, y=None, conditioning_x=None, **kwargs):
        y = (config.score_model.num_classes * torch.ones((config.sampling.batch_size,))).to(torch.long).to(dist_util.dev())
        return model(x, t, y = y, conditioning_x=conditioning_x)
    
    # Create an classifier-free guidance sampling function from Glide code
    def model_classifier_free_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + config.sampling.norm_cond_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps
    # classifier-free guidance without increasing batch - trading off space for time
    def model_classifier_free_opt_fn(x_t, ts, **kwargs):
        ## conditional diffusion output
        cond_eps = model(x_t, ts, **kwargs)
        ## unconditional diffusion output
        uncond_kwargs = kwargs.copy()
        uncond_kwargs["y"] = config.score_model.num_classes * torch.ones_like(kwargs["y"])
        uncond_eps = model(x_t, ts, **uncond_kwargs)
        eps = uncond_eps + config.sampling.norm_cond_scale * (cond_eps - uncond_eps)
        return eps
    
    def inpainting_denoised_fn(x_start,**kwargs):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * kwargs['inpaint_mask']
            + kwargs['image'] * (1 - kwargs['inpaint_mask'])
    )

    # dynamic normalisation
    def clamp_to_spatial_quantile(x : torch.Tensor, **kwargs):
        p = 0.99
        b, c, *spatial = x.shape
        quantile = torch.quantile(torch.abs(x).view(b,c,-1), p, dim = -1, keepdim =True)
        quantile = torch.max(quantile,torch.ones_like(quantile))
        quantile_broadcasted, _ = torch.broadcast_tensors(quantile.unsqueeze(-1),x)
        return torch.min(torch.max(x,-quantile_broadcasted), quantile_broadcasted) / quantile_broadcasted

    return cond_fn, model_fn, model_classifier_free_opt_fn, clamp_to_spatial_quantile



def get_dict_of_arrays(all_results):
    samples = {k: [get_numpy_from_torch(dic[k]) for dic in all_results] for k in all_results[0]}
    samples = {k:np.concatenate(v,0) for k,v in samples.items()}
    return samples

def get_numpy_from_torch(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        else:
            return tensor

def get_input_data(config, data_dict):

    model_kwargs = {k: v.to(dist_util.dev()) for k, v in data_dict.items()}
    # print('------233-----------')
    # print(model_kwargs["y"])    # tensor([0, 1, 1, 1, ..., 0, 0, 0], device='cuda:0')
    # print('------233 over-----------')

    for k, v in model_kwargs.items():
        print('k:{},type v:{}, size v:{}'.format(k, type(v), v.shape))
    # k:image,              type v:<class 'torch.Tensor'>,      size v:torch.Size([64, 4, 64, 64])
    # k:gt,                 type v:<class 'torch.Tensor'>,      size v:torch.Size([64, 1, 64, 64])
    # k:patient_id,         type v:<class 'torch.Tensor'>,      size v:torch.Size([64])
    # k:slice_id,           type v:<class 'torch.Tensor'>,      size v:torch.Size([64])
    # k:y,                  type v:<class 'torch.Tensor'>,      size v:torch.Size([64])
    # k:conditioning_x,     type v:<class 'torch.Tensor'>,      size v:torch.Size([64, 1, 64, 64])
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^
    # split:test, patient_id:2, slice_id:40
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^
    model_kwargs["y"] = (config.sampling.target_class * torch.ones((config.sampling.batch_size,))).to(torch.long).to(dist_util.dev())
    # print('------23333333-----------')
    # print(model_kwargs["y"])
    # print('------23333333 over, too-----------')

    init_image = data_dict['image'].to(dist_util.dev())

    return model_kwargs,init_image
