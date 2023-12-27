"""
Train a diffusion model on images.
"""
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import argparse
from baseline_code.configs import get_config
from baseline_code.utils import logger, dist_util
from baseline_code.models.resample import create_named_schedule_sampler
from baseline_code.utils.script_util import create_gaussian_diffusion, create_score_model
from baseline_code.training.train_util import TrainLoop
from baseline_code.datasets import loader
import torch as th


def main(args):
    config = get_config.file_from_dataset(args.dataset)
    print('+++++++config++++++++++')
    print(config)
    print('+++++++++++++++++++++++')

    dist_util.setup_dist()  # 设置一个分布式进程组。
    logger.configure(Path(config.experiment_name)/"score_train",
                     format_strs=["log", "stdout", "csv", "tensorboard"])

    logger.log("creating model and diffusion...")
    diffusion = create_gaussian_diffusion(config)       # 初始化一个gaussian扩散模型
    model = create_score_model(config)                  # 初始化一个score扩散模型
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # model.to(dist_util.dev())
    model = th.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    logger.log(f"Model number of parameters {pytorch_total_params}")    # Model number of parameters 27144708

    # show model size
    print('---------- Networks initialized -------------')
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print(model)
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    print('-----------------------------------------------')

    # schedule_sampler代表训练的时刻t时，NN的输入数据是以什么分布出现的(均匀采样or重要性采样)
    schedule_sampler = create_named_schedule_sampler(config.score_model.training.schedule_sampler, diffusion)
    # schedule_sampler="uniform"
    # config.classifier.training.noised is True ---记得在哪输出的


    logger.log("creating data loader...")   # dataset = "MICCAI_BraTS2020"
    train_loader = loader.get_data_loader(args.dataset, config, split_set='train', generator = True) 
    val_loader = loader.get_data_loader(args.dataset, config, split_set='val', generator = True)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_loader,
        data_val=val_loader,
        batch_size=config.score_model.training.batch_size,
        microbatch=config.score_model.training.microbatch,
        lr=config.score_model.training.lr,
        ema_rate=config.score_model.training.ema_rate,
        log_interval=config.score_model.training.log_interval,
        save_interval=config.score_model.training.save_interval,
        resume_checkpoint=config.score_model.training.resume_checkpoint,
        use_fp16=config.score_model.training.use_fp16,
        fp16_scale_growth=config.score_model.training.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=config.score_model.training.weight_decay,
        lr_anneal_steps=config.score_model.training.lr_anneal_steps,
        cond_dropout_rate = config.score_model.training.cond_dropout_rate if config.score_model.class_cond else 0,
        conditioning_variable = config.score_model.training.conditioning_variable,
        iterations = config.score_model.training.iterations
    ).run_loop()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="mnist or brats", type=str,default='brats')
    args = parser.parse_args()
    # print(args.dataset)
    main(args)


# cd /chenxue/Diff-SCM-main && python -u diff_scm/training/main_diffusion_train.py
# cd /chenxue/Diff-SCM-main && python -u diff_scm/sampling/sampling_counterfactual.py
# cd /chenxue/Diff-SCM-main && python -u diff_scm/sampling/sample_counterfactual.py
# cd /chenxue/Diff-SCM-main/diff_scm/datasets && python -u data_preprocessing.py
# cd /chenxue/Diff-SCM-main/diff_scm/datasets && mpiexec -n 4 python data_preprocessing.py
# cd /chenxue/Diff-SCM-main/diff_scm/sampling && mpiexec -n 4 python sampling_counterfactual.py
