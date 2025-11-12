# lumina_controlnet_2.5pro.py

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import toml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from tqdm import tqdm
from PIL import Image
import numpy as np
import functools

from library import (
    config_util,
    deepspeed_utils,
    lumina_models,
    lumina_train_util,
    lumina_util,
    strategy_base,
    strategy_lumina,
    train_util,
)
from library.device_utils import clean_memory_on_device, init_ipex
from library.flux_models import AutoEncoder
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler
from library.utils import add_logging_arguments, setup_logging
from transformers import Gemma2Model

# --- FIX: Suppress TensorFlow informational messages ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=info, 2=warning, 3=error
# --- END FIX ---


# 初始化IPEX和日志
init_ipex()
setup_logging()
import logging

logger = logging.getLogger(__name__)

# --- 从 lumina_train_util.py 适配的采样函数 ---
# 为了支持ControlNet，我们需要修改采样逻辑，在生成过程中注入控制信号

@torch.no_grad()
def denoise_controlnet(
    scheduler: FlowMatchEulerDiscreteScheduler,
    base_model: lumina_models.NextDiT,
    controlnet: lumina_models.ControlNetLumina,
    img: Tensor,
    txt: Tensor,
    txt_mask: Tensor,
    neg_txt: Tensor,
    neg_txt_mask: Tensor,
    controlnet_cond: Tensor,
    timesteps: Union[List[float], torch.Tensor],
    guidance_scale: float = 4.0,
    cfg_trunc_ratio: float = 0.25,
    renorm_cfg: float = 1.0,
):
    """
    使用基础模型和ControlNet对图像进行降噪。
    """
    for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
        # Lumina使用 t=0 表示噪声, t=1 表示图像, 因此反转时间步
        current_timestep = 1 - t / scheduler.config.num_train_timesteps
        current_timestep = current_timestep * torch.ones(img.shape[0], device=img.device)

        # 1. 从ControlNet获取控制残差
        controlnet_residuals, _ = controlnet(
            x=img,
            t=current_timestep,
            cap_feats=txt,
            cap_mask=txt_mask.to(dtype=torch.int32),
            controlnet_cond=controlnet_cond,
        )

        # 2. 将残差注入基础模型进行条件预测
        noise_pred_cond = base_model(
            x=img,
            t=current_timestep,
            cap_feats=txt,
            cap_mask=txt_mask.to(dtype=torch.int32),
            controlnet_residuals=controlnet_residuals,
        )

        # CFG
        if current_timestep[0] < cfg_trunc_ratio:
            # 无条件预测时也需要从ControlNet获取残差（尽管对于某些类型的ControlNet可能影响不大）
            uncond_controlnet_residuals, _ = controlnet(
                x=img,
                t=current_timestep,
                cap_feats=neg_txt,
                cap_mask=neg_txt_mask.to(dtype=torch.int32),
                controlnet_cond=controlnet_cond,
            )
            noise_pred_uncond = base_model(
                x=img,
                t=current_timestep,
                cap_feats=neg_txt,
                cap_mask=neg_txt_mask.to(dtype=torch.int32),
                controlnet_residuals=uncond_controlnet_residuals,
            )
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # 重新归一化
            if float(renorm_cfg) > 0.0:
                cond_norm = torch.linalg.vector_norm(noise_pred_cond, dim=tuple(range(1, len(noise_pred_cond.shape))), keepdim=True)
                max_new_norms = cond_norm * float(renorm_cfg)
                noise_norms = torch.linalg.vector_norm(noise_pred, dim=tuple(range(1, len(noise_pred.shape))), keepdim=True)
                for j, (noise_norm, max_new_norm) in enumerate(zip(noise_norms, max_new_norms)):
                    if noise_norm >= max_new_norm:
                        noise_pred[j] = noise_pred[j] * (max_new_norm / noise_norm)
        else:
            noise_pred = noise_pred_cond

        # 计算上一步的带噪样本 x_t -> x_t-1
        noise_pred = -noise_pred
        img = scheduler.step(noise_pred, t, img, return_dict=False)[0]

    return img


@torch.no_grad()
def sample_images_controlnet(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    base_model: lumina_models.NextDiT,
    controlnet: lumina_models.ControlNetLumina,
    vae: AutoEncoder,
    gemma2_model: Gemma2Model,
    sample_prompts_gemma2_outputs: dict,
):
    """
    使用ControlNet生成样本图像。
    """
    logger.info(f"Generating sample images with ControlNet at step: {global_step}")

    if not args.sample_prompts:
        logger.warning("No sample prompts provided.")
        return

    # 解包模型
    base_model = accelerator.unwrap_model(base_model)
    controlnet = accelerator.unwrap_model(controlnet)
    if gemma2_model:
        gemma2_model = accelerator.unwrap_model(gemma2_model)
    
    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # 保存和恢复随机状态
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    # 初始化策略
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    for prompt_dict in prompts:
        prompt = prompt_dict.get("prompt", "")
        negative_prompt = prompt_dict.get("negative_prompt", "")
        controlnet_image_path = prompt_dict.get("controlnet_image")

        if not controlnet_image_path or not os.path.exists(controlnet_image_path):
            logger.warning(f"ControlNet image not found for prompt: '{prompt}'. Skipping.")
            continue

        # 参数设定
        width = int(prompt_dict.get("width", 1024))
        height = int(prompt_dict.get("height", 1024))
        guidance_scale = float(prompt_dict.get("scale", 3.5))
        sample_steps = int(prompt_dict.get("sample_steps", 36))
        seed = int(prompt_dict.get("seed", time.time()))

        generator = torch.Generator(device=accelerator.device).manual_seed(seed)

        # 准备控制图像
        control_image = Image.open(controlnet_image_path).convert("RGB")
        control_image = control_image.resize((width, height), Image.LANCZOS)
        control_image = np.array(control_image).astype(np.float32) / 255.0
        control_image = torch.from_numpy(control_image).permute(2, 0, 1).unsqueeze(0)
        controlnet_cond = control_image.to(accelerator.device, dtype=vae.dtype)

        # 文本编码
        system_prompt = f"{args.system_prompt} <Prompt Start> " if args.system_prompt else ""
        
        # 条件编码
        if prompt in sample_prompts_gemma2_outputs:
            cond_hidden, _, cond_mask = sample_prompts_gemma2_outputs[prompt]
        else:
            tokens = tokenize_strategy.tokenize(system_prompt + prompt)
            cond_hidden, _, cond_mask = encoding_strategy.encode_tokens(tokenize_strategy, [gemma2_model], tokens)
        
        # 无条件编码
        if negative_prompt in sample_prompts_gemma2_outputs:
            uncond_hidden, _, uncond_mask = sample_prompts_gemma2_outputs[negative_prompt]
        else:
            tokens = tokenize_strategy.tokenize(negative_prompt)
            uncond_hidden, _, uncond_mask = encoding_strategy.encode_tokens(tokenize_strategy, [gemma2_model], tokens)
        
        # 准备潜空间噪声
        latents_shape = (1, 16, height // 8, width // 8)
        noise = torch.randn(latents_shape, generator=generator, device=accelerator.device, dtype=vae.dtype)

        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.0)
        timesteps, _ = lumina_train_util.retrieve_timesteps(scheduler, num_inference_steps=sample_steps)

        # 执行降噪
        with accelerator.autocast():
            denoised_latents = denoise_controlnet(
                scheduler,
                base_model,
                controlnet,
                noise,
                cond_hidden.to(accelerator.device, dtype=vae.dtype),
                cond_mask.to(accelerator.device),
                uncond_hidden.to(accelerator.device, dtype=vae.dtype),
                uncond_mask.to(accelerator.device),
                controlnet_cond,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
            )

        # VAE解码
        vae.to(accelerator.device)
        with accelerator.autocast():
            img = denoised_latents / vae.scale_factor + vae.shift_factor
            img = vae.decode(img)
        vae.to("cpu")
        
        img = img.clamp(-1, 1).permute(0, 2, 3, 1).float().cpu().numpy()
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        
        # 保存图像
        image = Image.fromarray(img[0])
        ts_str = time.strftime("%Y%m%d%H%M%S")
        step_str = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
        filename = f"{args.output_name or 'sample'}_{step_str}_{ts_str}_{seed}.png"
        image.save(os.path.join(save_dir, filename))

    # 恢复随机状态
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    clean_memory_on_device(accelerator.device)



def initialize_worker_strategies(worker_id, gemma2_max_token_length):
    """
    确保在每个数据加载子进程中都正确设置了分词和编码策略。
    这是一个顶层函数，以便能够被多进程 DataLoader 'pickle'。
    """
    # 这个函数将在每个 worker 进程启动时被调用
    strategy_base.TokenizeStrategy.set_strategy(
        strategy_lumina.LuminaTokenizeStrategy(gemma2_max_token_length)
    )
    strategy_base.TextEncodingStrategy.set_strategy(strategy_lumina.LuminaTextEncodingStrategy())


# --- 适配的模型保存函数 ---

def simple_collator(batch):
    """
    一个简单的 collate 函数，返回批次中的第一个元素。
    由于数据集每次只返回一个样本字典，这个函数的作用是解包。
    """
    return batch[0]

def save_controlnet_model_on_train_end(
    args: argparse.Namespace, save_dtype: torch.dtype, epoch: int, global_step: int, controlnet: nn.Module
):
    def sd_saver(ckpt_file, _, __):
        sai_metadata = train_util.get_sai_model_spec(None, args, False, False, True, lumina="lumina2_controlnet")
        lumina_train_util.save_models(ckpt_file, controlnet, sai_metadata, save_dtype, args.mem_eff_save)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, "controlnet")


def save_controlnet_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace, on_epoch_end: bool, accelerator: Accelerator, save_dtype: torch.dtype, 
    epoch: int, num_train_epochs: int, global_step: int, controlnet: nn.Module
):
    def sd_saver(ckpt_file, _, __):
        sai_metadata = train_util.get_sai_model_spec(None, args, False, False, True, lumina="lumina2_controlnet")
        lumina_train_util.save_models(ckpt_file, controlnet, sai_metadata, save_dtype, args.mem_eff_save)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args, on_epoch_end, accelerator, True, True, epoch, num_train_epochs, 
        global_step, sd_saver, "controlnet"
    )

# --- 主训练函数 ---

def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)


    if args.seed is not None:
        set_seed(args.seed)

    # --- 1. 准备Accelerator和模型 ---
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # --- 2. 加载数据处理所需的模型（VAE, 文本编码器） ---
    ae = lumina_util.load_ae(args.ae, weight_dtype, "cpu") if args.cache_latents or not args.cache_text_encoder_outputs else None
    gemma2 = lumina_util.load_gemma2(args.gemma2, weight_dtype, "cpu")
    gemma2.eval().requires_grad_(False)

    # strategy_base.TokenizeStrategy.set_strategy(
    #     strategy_lumina.LuminaTokenizeStrategy(args.gemma2_max_token_length)
    # )
    # strategy_base.TextEncodingStrategy.set_strategy(strategy_lumina.LuminaTextEncodingStrategy())

    # --- 3. 准备和缓存数据集 ---
    if args.cache_latents:
        strategy_base.LatentsCachingStrategy.set_strategy(
            strategy_lumina.LuminaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)
        )
        logger.info("Caching latents...")
        if ae is not None:
            ae.to(accelerator.device, dtype=weight_dtype)
            ae.to("cpu")
            clean_memory_on_device(accelerator.device)
    
    sample_prompts_gemma2_outputs = {}
    if args.cache_text_encoder_outputs:
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
            strategy_lumina.LuminaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check
            )
        )
        logger.info("Caching text encoder outputs...")
        gemma2.to(accelerator.device, dtype=weight_dtype)
        if args.sample_prompts:
            prompts = train_util.load_prompts(args.sample_prompts)
            for p_dict in prompts:
                for p_key in ["prompt", "negative_prompt"]:
                    p = p_dict.get(p_key, "")
                    if p and p not in sample_prompts_gemma2_outputs:
                        tokens = strategy_base.TokenizeStrategy.get_strategy().tokenize(p)
                        outputs = strategy_base.TextEncodingStrategy.get_strategy().encode_tokens(None, [gemma2], tokens)
                        sample_prompts_gemma2_outputs[p] = [o.cpu() for o in outputs]
        gemma2.to("cpu")
        clean_memory_on_device(accelerator.device)

    # 创建数据集
    if args.dataset_class is None:
        blueprint_generator = config_util.BlueprintGenerator(config_util.ConfigSanitizer(False, False, True, True))
        user_config = { "datasets": [{ "subsets": config_util.generate_controlnet_subsets_config_by_subdirs(
            args.train_data_dir, args.conditioning_data_dir, args.caption_extension
        )}]} if not args.dataset_config else config_util.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)

    if accelerator.is_main_process:
        logger.info(f"Total number of dataset images found: {len(train_dataset_group)}")

    if len(train_dataset_group) == 0:
        logger.error(
            "Dataset is empty. Please check your folder structure and arguments. "
            "Ensure that --train_data_dir and --conditioning_data_dir have matching filenames "
            "and that caption files with the specified --caption_extension exist."
        )
        return  # 如果没有找到数据则退出

    # 对创建好的数据集执行缓存操作
    # if args.cache_latents:
    #     train_dataset_group.new_cache_latents(ae, accelerator, vae_batch_size=args.vae_batch_size)
    # 修改后的代码 (正确)
    if args.cache_latents:
        train_dataset_group.new_cache_latents(ae, accelerator)
    if args.cache_text_encoder_outputs:
        train_dataset_group.new_cache_text_encoder_outputs([gemma2], accelerator)

    # --- 4. 加载训练模型和创建数据加载器 ---
    #collator = lambda x: x[0]
    
    # 为DataLoader的每个工作进程定义一个初始化函数
    def worker_init_fn(worker_id):
        """
        确保在每个数据加载子进程中都正确设置了分词和编码策略。
        """
        strategy_base.TokenizeStrategy.set_strategy(
            strategy_lumina.LuminaTokenizeStrategy(args.gemma2_max_token_length)
        )
        strategy_base.TextEncodingStrategy.set_strategy(strategy_lumina.LuminaTextEncodingStrategy())

    worker_init_fn_partial = functools.partial(
        initialize_worker_strategies, gemma2_max_token_length=args.gemma2_max_token_length
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=simple_collator,
        #num_workers=args.max_data_loader_n_workers,
        num_workers=0,
        persistent_workers=args.persistent_data_loader_workers,
        worker_init_fn=worker_init_fn_partial,  # <-- 使用新的 partial 函数
    )

    # 加载Lumina基础模型 (冻结) - 确保它在CPU上以便后续权重复制
    logger.info(f"Loading base Lumina model from {args.pretrained_model_name_or_path}")
    base_model = lumina_util.load_lumina_model(
        args.pretrained_model_name_or_path, weight_dtype, "cpu", use_flash_attn=args.use_flash_attn
    )
    base_model.requires_grad_(False)
    base_model.eval()

    # ##################################################################
    # #                           【最终修复】                         #
    # ##################################################################
    # 加载或初始化ControlNet (可训练)
    logger.info(f"Loading ControlNet model from {args.controlnet_model_name_or_path}")

    # 1. 调用函数获取在 "meta" 设备上定义的ControlNet模型结构
    # 这一步返回的模型包含占位符张量，无法直接移动
    controlnet_with_meta_tensors = lumina_util.load_controlnet(
        args.controlnet_model_name_or_path, weight_dtype, "cpu", base_model=base_model
    )

    # 2. 使用 .to_empty() 将模型结构“物化”到CPU上。
    # 这会创建具有正确形状和数据类型的真实张量，但权重是未初始化的。
    controlnet = controlnet_with_meta_tensors.to_empty(device="cpu")

    # 3. 将base_model的权重加载到新创建的、空的ControlNet中。
    # `strict=False` 至关重要，因为它只复制名称匹配的层的权重。
    # 这会正确地从基础模型初始化ControlNet的共有部分，
    # 而ControlNet独有的层（例如zero-convs）将保留其初始状态并从头开始训练。
    controlnet.load_state_dict(base_model.state_dict(), strict=False)

    # 至此, `controlnet` 是一个在CPU上被正确初始化的完整模型。
    # ##################################################################
    # #                           【修复结束】                         #
    # ##################################################################
    
    # 现在可以安全地将 base_model 移动到目标设备了
    base_model.to(accelerator.device)
    
# ========================= DEFINITIVE FIX START =========================
    # 1. 确保 ControlNet 在正确的设备和数据类型上
    logger.info(f"Moving ControlNet to device: {accelerator.device} with dtype: {weight_dtype}")
    controlnet = controlnet.to(accelerator.device, dtype=weight_dtype)

    # 2. 核心修复：显式地将 ControlNet 的所有参数设置为可训练。
    #    这是解决 "empty parameter list" 错误的关键，因为它能防止
    #    优化器函数因所有参数都被冻结而过滤掉它们。
    logger.info("Setting requires_grad=True for all ControlNet parameters.")
    controlnet.requires_grad_(True)
    
    # 3. 将模型设置为训练模式
    controlnet.train()
    # ========================== DEFINITIVE FIX END ==========================

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    if not args.cache_latents:
        if ae is None: # if not loaded before
             ae = lumina_util.load_ae(args.ae, weight_dtype, "cpu")
        ae.to(accelerator.device, dtype=weight_dtype)
        ae.eval().requires_grad_(False)

    if not args.cache_text_encoder_outputs:
        gemma2.to(accelerator.device, dtype=weight_dtype)

    # --- 5. 准备优化器和学习率调度器 ---
    # params_to_optimize = controlnet.parameters()
    # ========================= FINAL FIX =========================
    # 核心修复：将 controlnet.parameters() 返回的生成器(generator)显式转换为列表(list)。
    # 这是因为 bitsandbytes 的 AdamW8bit 优化器可能需要多次迭代参数列表，
    # 而生成器只能被迭代一次。转换为列表解决了这个兼容性问题。
    logger.info("Converting ControlNet parameters generator to a list for the optimizer.")
    params_to_optimize = list(controlnet.parameters())
    # =============================================================
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
    
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)
    
    # --- 6. Accelerator `prepare` ---
    # 此时 controlnet 已经位于正确的设备上，prepare 不会再尝试移动它
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # --- 7. 训练循环 ---
    noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=args.discrete_flow_shift)
    
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    if accelerator.is_main_process:
        accelerator.init_trackers("lumina-controlnet")

    for epoch in range(args.max_train_epochs or 1):
        for step, batch in enumerate(train_dataloader):
            if accelerator.is_main_process:
                train_dataset_group.set_current_epoch(epoch)
            with accelerator.accumulate(controlnet):
                if accelerator.is_main_process:
                    train_dataset_group.set_current_step(global_step)
                
                with torch.no_grad():
                    # 修复：增加对 batch["latents"] is None 的检查
                    latents = ae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)) if "latents" not in batch or batch["latents"] is None else batch["latents"].to(accelerator.device, dtype=weight_dtype)

                if "text_encoder_outputs_list" in batch and batch["text_encoder_outputs_list"] is not None:
                    cap_feats, _, cap_mask = batch["text_encoder_outputs_list"]
                    cap_feats = cap_feats.to(accelerator.device, dtype=weight_dtype)
                    cap_mask = cap_mask.to(accelerator.device)
                else:
                    with torch.no_grad():
                        # 从 batch 中获取 token IDs
                        tokens = batch["input_ids_list"]
                        # 使用 gemma2 模型实时编码
                        # 注意：gemma2的输出结构可能需要确认，这里假设它返回一个包含 hidden_states 的元组
                        encoder_outputs = gemma2(
                            input_ids=tokens[0].to(accelerator.device), 
                            attention_mask=tokens[1].to(accelerator.device), 
                            output_hidden_states=True
                        )
                        # Lumina 通常使用倒数第二层的隐藏状态作为特征
                        cap_feats = encoder_outputs.hidden_states[-2]
                        # cap_mask 应该从 tokens 中获取
                        cap_mask = tokens[1].to(accelerator.device)

                noise = torch.randn_like(latents)
                
                noisy_model_input, timesteps, sigmas = lumina_train_util.get_noisy_model_input_and_timesteps(
                    args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
                )

                with accelerator.autocast():
                    controlnet_residuals, _ = controlnet(
                        x=noisy_model_input,
                        t=timesteps,
                        cap_feats=cap_feats,
                        cap_mask=cap_mask.to(dtype=torch.int32),
                        controlnet_cond=batch["conditioning_images"].to(accelerator.device, dtype=weight_dtype),
                    )

                    # 移除 with torch.no_grad(): 块
                    # 尽管 base_model 被冻结, 我们仍需梯度流过它以到达 controlnet_residuals
                    model_pred = base_model(
                        x=noisy_model_input,
                        t=timesteps,
                        cap_feats=cap_feats,
                        cap_mask=cap_mask.to(dtype=torch.int32),
                        controlnet_residuals=controlnet_residuals,
                    )

                model_pred, weighting = lumina_train_util.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)
                target = noise - latents
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
                if weighting is not None:
                    loss *= weighting
                loss = loss.mean()
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step > 0 and global_step % args.save_every_n_steps == 0 and accelerator.is_main_process:
                    save_controlnet_model_on_epoch_end_or_stepwise(
                        args, False, accelerator, save_dtype, epoch, args.max_train_epochs, global_step, accelerator.unwrap_model(controlnet)
                    )

                if global_step > 0 and global_step % args.sample_every_n_steps == 0 and accelerator.is_main_process:
                    sample_images_controlnet(
                        accelerator, args, epoch, global_step, base_model, accelerator.unwrap_model(controlnet), ae, gemma2, sample_prompts_gemma2_outputs
                    )
            
            progress_bar.set_postfix(loss=loss.item())
            accelerator.log({"loss": loss.item()}, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_controlnet_model_on_train_end(args, save_dtype, epoch, global_step, accelerator.unwrap_model(controlnet))

    accelerator.end_training()
    logger.info("Training finished.")


# --- 参数解析器 ---
def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    
    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, False, True, True)
    train_util.add_training_arguments(parser, False)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    lumina_train_util.add_lumina_train_arguments(parser)

    # ========================= FIX START =========================
    # 添加缺失的 weighting_scheme 参数
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logsnr",
        choices=["logsnr", "min_snr", "default"],
        help="Weighting scheme for loss. 'logsnr' or 'min_snr' are common choices for flow-matching models.",
    )
    # 添加缺失的 logit_mean 参数
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="Mean for logit normal distribution for noise sampling schedule.",
    )
    # 添加缺失的 logit_std 参数 
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.1,
        help="Standard deviation for logit normal distribution for noise sampling schedule.",
    )
    # 添加缺失的 mode_scale 参数
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.25,
        help="Mode scale parameter for noise schedule.",
    )
    # ========================== FIX END ==========================

    parser.add_argument("--cache_text_encoder_outputs", action="store_true",
                        help="cache text encoder outputs to RAM or disk.")
    parser.add_argument("--cache_text_encoder_outputs_to_disk", action="store_true",
                        help="cache text encoder outputs to disk instead of RAM.")
    parser.add_argument("--text_encoder_batch_size", type=int, default=None, 
                        help="batch size for text encoder when caching.")
    
    parser.add_argument("--scale_v_pred_loss_like_noise_pred", action="store_true",
                        help="scale v prediction loss like noise prediction loss.")
    parser.add_argument("--v_pred_like_loss", action="store_true",
                        help="enable v-prediction like loss.")

    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None,
                        help="path to ControlNet model to continue training, or to init with random weights if not exists")
    
    parser.add_argument("--conditioning_data_dir", type=str, default=None,
                        help="directory for conditioning images for ControlNet. Must have the same folder structure as train_data_dir.")

    parser.add_argument("--mem_eff_save", action="store_true", help="use memory efficient saving")
    
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

        # 在主程序块中设置全局策略 (保留此部分)
    print("Setting global tokenization and text encoding strategies in main execution block...")
    from library import strategy_base, strategy_lumina
    strategy_base.TokenizeStrategy.set_strategy(
        strategy_lumina.LuminaTokenizeStrategy(args.gemma2_max_token_length)
    )
    strategy_base.TextEncodingStrategy.set_strategy(
        strategy_lumina.LuminaTextEncodingStrategy()
    )

    # ========================== 猴子补丁开始 ==========================
    # 错误反复出现，表明库内部存在复杂的初始化问题。
    # 我们通过猴子补丁在运行时动态修复 DreamBoothDataset 类。
    print("Applying monkey-patch to library.train_util.DreamBoothDataset...")
    from library import train_util

    # 1. 保存对原始 __getitem__ 方法的引用
    original_dreambooth_getitem = train_util.DreamBoothDataset.__getitem__

    # 2. 定义我们的补丁函数
    def patched_dreambooth_getitem(self, index):
        # 核心修复：在调用原始方法前，检查并修复 tokenize_strategy 属性
        if self.tokenize_strategy is None:
            # print("Monkey-patch activated: Found None strategy, injecting global strategy.")
            self.tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        
        # 3. 调用原始的 __getitem__ 方法，此时 self.tokenize_strategy 已被修复
        return original_dreambooth_getitem(self, index)

    # 4. 用我们的补丁函数替换掉类中原始的方法
    train_util.DreamBoothDataset.__getitem__ = patched_dreambooth_getitem
    print("Monkey-patch applied successfully.")
    # =========================== 猴子补丁结束 ===========================

    # 最后，调用 train 函数

    train(args)
