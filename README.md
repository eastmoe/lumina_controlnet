## Lumina Controlnet训练脚本

这是一个未完工的脚本，基于[sdbds/sd-scripts](https://github.com/sdbds/sd-scripts/tree/lumina)（一个[Kohya_ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)的支持Lumina的分支）修改。当前未完工，也未经测试。


#### 准备数据集

```✨ Lumina2Dataset (数据集)
┃
┠───📂 image (文件夹1: 图像与灵感)
┃   ┃
┃   ┠── 🎨 1.png  (一张美丽的画)
┃   ┠── 📝 1.txt  ("一只猫咪坐在窗边，望着星空")
┃   ┃
┃   ┠── 🎨 2.png  (另一张奇妙的图)
┃   ┠── 📝 2.txt  ("赛博朋克城市的雨夜，霓虹灯闪烁")
┃   ┃
┃   ┠── 🎨 3.png  (......)
┃   ┠── 📝 3.txt  (......)
┃   ┃
┃   ┖── ... (更多成对的图像和描述)
┃
┠───📂 condition (文件夹2: 结构与引导)
┃   ┃
┃   ┠── 📐 1.png  (与画1对应的结构图)
┃   ┠── 📐 2.png  (与画2对应的结构图)
┃   ┠── 📐 3.png  (......)
┃   ┖── ... (更多与图像一一对应的引导图)
┃
┖── ... (其他文件夹)
```

#### 使用步骤；
1、部署好kohya_ss环境。

2、按照目录结构，把文件复制到kohya_ss的sd-scripts目录下。

3、下载Lumina diffusion model、CLIP（gemma2-2b）、VAE（flux的ae）。

4、修改命令中的各种参数。

5、在终端里进入kohya_ss的虚拟环境，并进入sd-scripts目录，执行训练命令。

**（下面的命令只是演示格式，其中存在一些错误或不当的部分）**

```python lumina_controlnet_2.5pro.py      --console_log_level INFO     --console_log_file "D:\\kohya_ss\\logs\\train\\lumina2.log"     --pretrained_model_name_or_path    "D:\\LuminaCtTrain\\models\\diffusion_model\\NetaYumev35_pretrained_unet.safetensors"     --gemma2 "D:\\LuminaCtTrain\\models\\text_encoder\\gemma_2_2b_fp16.safetensors" --tokenizer_cache_dir "D:\\LuminaCtTrain\\models\\text_encoder"   --gemma2_max_token_length 4096     --ae "D:\\LuminaCtTrain\\models\\vae\\flux_ae.safetensors"     --train_data_dir "D:\\kohya_ss\\dataset\\Lumina2Dataset\\image"     --conditioning_data_dir "D:\\kohya_ss\\dataset\\Lumina2Dataset\\deepth"         --resolution 1024     --train_batch_size 1     --caption_extension "txt"     --output_dir  "D:\\kohya_ss\\outputs\\lumina2Deepth"     --output_name "lumina2_deepth_controlnet"     --save_every_n_steps 100     --xformers     --sdpa     --max_train_steps 20000     --seed 23672323     --mixed_precision fp16     --full_fp16  --gradient_checkpointing    --clip_skip 2     --metadata_author "星月StarMoon"     --fp16_master_weights_and_gradients     --optimizer_type AdamW8bit     --learning_rate 5e-6     --lr_scheduler cosine     --lr_warmup_steps 100     --save_model_as safetensors```


#### 命令说明：

* 预训练UNet模型: D:\LuminaCtTrain\models\diffusion_model\NetaYumev35_pretrained_unet.safetensors
    - --pretrained_model_name_or_path

* Gemma2 文本编码器: D:\LuminaCtTrain\models\text_encoder\gemma_2_2b_fp16.safetensors
    - --gemma2

* Tokenizer 缓存目录: D:\LuminaCtTrain\models\text_encoder
    - --tokenizer_cache_dir
* AE (Autoencoder) 模型: D:\LuminaCtTrain\models\vae\flux_ae.safetensors
    - --ae
* 训练图像数据: D:\kohya_ss\dataset\Lumina2Dataset\image
    - --train_data_dir
* 条件图像数据 (深度图): D:\kohya_ss\dataset\Lumina2Dataset\deepth
    - --conditioning_data_dir
* 输出目录: D:\kohya_ss\outputs\lumina2Deepth
    - --output_dir
* 日志文件: D:\kohya_ss\logs\train\lumina2.log
    - --console_log_file
* 分辨率: 1024
    - --resolution
* 批次大小 (Batch Size): 1
    - --train_batch_size
* 最大训练步数: 20000
    - --max_train_steps
* 文本标签文件扩展名: txt
    - --caption_extension
* Gemma2 最大Token长度: 4096
    - --gemma2_max_token_length
* Clip Skip: 2
    - --clip_skip
* 随机种子: 23672323
- --seed

* 输出模型名称: lumina2_deepth_controlnet
    - --output_name
* 模型保存格式: safetensors
    - --save_model_as
* 保存频率: 每 100 步保存一次
- --save_every_n_steps
* 元数据作者: 星月StarMoon
- --metadata_author

* 混合精度: fp16
    - --mixed_precision
* 启用 xformers: 是
- --xformers
* 启用 SDPA (Scaled Dot Product Attention): 是
- --sdpa
* 启用完整 FP16: 是
- --full_fp16
* 启用梯度检查点: 是
- --gradient_checkpointing
* 启用 FP16 主权重和梯度: 是
- --fp16_master_weights_and_gradients

* 优化器类型: AdamW8bit
    - --optimizer_type
* 学习率: 5e-6 (即 0.000005)
    - --learning_rate
* 学习率调度器: cosine (余弦退火)
    - --lr_scheduler
* 学习率预热步数: 100
    - --lr_warmup_steps

* 控制台日志级别: INFO
    - --console_log_level

### 致谢
- [sdbds/sd-scripts](https://github.com/sdbds/sd-scripts/tree/lumina)
- [Kohya_ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [Gemini 2.5 Pro in Google AI Studio](https://aistudio.google.com/prompts/new_chat)
