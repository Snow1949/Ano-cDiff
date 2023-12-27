import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.训练时的原始扩散步数
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N(均匀划分成N部分)
                           is a number of steps to use the striding from the
                           DDIM paper.采样时分几部分来采样
    :return: a set of diffusion steps from the original process to use.
    返回一个新的时间序列,从num_timesteps中根据section_counts取得
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    一种扩散过程，它可以跳过基本扩散过程中的步骤,
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    保留的原始扩散过程的时间步长的集合(序列或集合
    :param kwargs: the kwargs to create the base diffusion process.创建基本扩散过程的kwargs。
    """
    # 定义加噪方案
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)     # 即可用的时间步长,if==1(原始间隔采样),if>1(respacing)
        self.timestep_map = []      # 基本等同于use_timesteps,不过是列表形式，连续的还是spacing
        self.original_num_steps = len(kwargs["betas"])  # 原始步长，做多少步加噪

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa 实例化基础扩散GaussianDiffusion
        last_alpha_cumprod = 1.0

        # 重新定义加噪方案betas序列(计算全新采样时刻后的betas)
        new_betas = []      # 对基础扩散中的阿尔法(alphas_cumprod)进行遍历
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:     # 如果索引i存在于新的时间序列中,
                # 来自beta与alpha之间的关系式
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)        # 将当前阿尔法包含进来,
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)         # 同时将i也放入新的时间表中
        #print(self.timestep_map)
        print(self.timestep_map)            # [0, 10, 20, 30, 40, 50, 60, ..., 970, 980, 990]

        # 更新父类self.betas成员变量为respacing对应的new_betas列表
        print('new_betas:{}'.format(new_betas))
        kwargs["betas"] = np.array(new_betas)       # 此处更新了betas
        super().__init__(**kwargs)

    def p_mean_variance(        # 用模型输出得到t-1时刻的均值和方差,同时预测X0。p就是NN预测出来的分布
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(            # 超参数不用，会得到不同形式的目标函数的公式
        self, model, *args, **kwargs        # 参数传入后,模型先过一遍包裹函数_wrap_model
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):        # 如果model已经是_WrappedModel类型实例,直接返回model
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    # 对model,当前时间步长图,做respacing的时间图以及原始时间图重新实例化为_WrappedModel,
    # 用新的实例化为_WrappedModel类型的model代替原来的model

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    """
    包裹模型进行一下后处理
    """
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps      # 一般固定在0-1000以内
        self.original_num_steps = original_num_steps    # 训练时用了多少步

    def __call__(self, x, ts, **kwargs):
        # ts是连续的索引，map_tensor中包含的是spacing后的索引
        # __call__的作用是将ts映射到真正的spacing后的时间步骤
        # 将timestep_map放到指定设备上(引入新变量一定要与老变量在同一设备上)
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            # 始终控制new_ts在[0,1000]以内的浮点数
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)
