import os
import sys
import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

# 批处理配置
testset_dir = "path to testset_dir"
base_save_path = "test_geoworld_stage2"
base_stage1_path = "test_geoworld_stage1"

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), 
                 os.path.dirname(os.path.dirname(current_file_path)), 
                 os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                               WanT5EncoderModel)
from videox_fun.models import WanTransformer3DModelGeoWorldstage2 as WanTransformer3DModel
from videox_fun.data.dataset_image_video import process_pose_file
from videox_fun.models.cache_utils import get_teacache_coefficients
from videox_fun.pipeline import WanFunControlPipelineGeoWorldstage2 as WanFunControlPipeline
from videox_fun.utils.fp8_optimization import (convert_model_weight_to_float8,
                                               convert_weight_dtype_wrapper,
                                               replace_parameters_by_name)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import (filter_kwargs, get_image_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from vggt.models.vggt import VGGT
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import torch.nn.functional as F

# GPU memory mode, which can be choosen in [model_full_load, model_full_load_and_qfloat8, model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
# model_full_load means that the entire model will be moved to the GPU.
# 
# model_full_load_and_qfloat8 means that the entire model will be moved to the GPU,
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
# 
# model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use, 
# and the transformer model has been quantized to float8, which can save more GPU memory. 
# 
# sequential_cpu_offload means that each layer of the model will be moved to the CPU after use, 
# resulting in slower speeds but saving a large amount of GPU memory.
GPU_memory_mode     = ""#"sequential_cpu_offload"
# Multi GPUs config
# Please ensure that the product of ulysses_degree and ring_degree equals the number of GPUs used. 
# For example, if you are using 8 GPUs, you can set ulysses_degree = 2 and ring_degree = 4.
# If you are using 1 GPU, you can set ulysses_degree = 1 and ring_degree = 1.
ulysses_degree      = 1
ring_degree         = 1
# Use FSDP to save more GPU memory in multi gpus.
fsdp_dit            = False
fsdp_text_encoder   = True
# Compile will give a speedup in fixed resolution and need a little GPU memory. 
# The compile_dit is not compatible with the fsdp_dit and sequential_cpu_offload.
compile_dit         = False

# Support TeaCache.
enable_teacache     = True
# Recommended to be set between 0.05 and 0.30. A larger threshold can cache more steps, speeding up the inference process, 
# but it may cause slight differences between the generated content and the original content.
# # --------------------------------------------------------------------------------------------------- #
# | Model Name          | threshold | Model Name          | threshold | Model Name          | threshold |
# | Wan2.1-T2V-1.3B     | 0.05~0.10 | Wan2.1-T2V-14B      | 0.10~0.15 | Wan2.1-I2V-14B-720P | 0.20~0.30 |
# | Wan2.1-I2V-14B-480P | 0.20~0.25 | Wan2.1-Fun-*-1.3B-* | 0.05~0.10 | Wan2.1-Fun-*-14B-*  | 0.20~0.30 |
# # --------------------------------------------------------------------------------------------------- #
teacache_threshold  = 0.10
# The number of steps to skip TeaCache at the beginning of the inference process, which can
# reduce the impact of TeaCache on generated video quality.
num_skip_start_steps = 5
# Whether to offload TeaCache tensors to cpu to save a little bit of GPU memory.
teacache_offload    = False

# Skip some cfg steps in inference
# Recommended to be set between 0.00 and 0.25
cfg_skip_ratio      = 0

# Riflex config
enable_riflex       = False
# Index of intrinsic frequency
riflex_k            = 6

# Config and model path
config_path         = "config/wan2.1/wan_civitai.yaml"
# model path
model_name          = "models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-Control"

# Choose the sampler in "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name        = "Flow"
# [NOTE]: Noise schedule shift parameter. Affects temporal dynamics. 
# Used when the sampler is in "Flow_Unipc", "Flow_DPM++".
# If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
# If you want to generate a 720p video, it is recommended to set the shift value to 5.0.
shift               = 3 

# Load pretrained model if need
transformer_path    = None
vae_path            = None
lora_path           = None

# Other params
sample_size         = [480, 720] # 20250721 # previous: [480, 720]
video_length        = 49
fps                 = 8

# Use torch.float16 if GPU does not support torch.bfloat16
# ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
weight_dtype            = torch.bfloat16

# 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
# 在neg prompt中添加"安静，固定"等词语可以增加动态性。
#prompt              = "在这个阳光明媚的户外花园里，美女身穿一袭及膝的白色无袖连衣裙，裙摆在她轻盈的舞姿中轻柔地摆动，宛如一只翩翩起舞的蝴蝶。阳光透过树叶间洒下斑驳的光影，映衬出她柔和的脸庞和清澈的眼眸，显得格外优雅。仿佛每一个动作都在诉说着青春与活力，她在草地上旋转，裙摆随之飞扬，仿佛整个花园都因她的舞动而欢愉。周围五彩缤纷的花朵在微风中摇曳，玫瑰、菊花、百合，各自释放出阵阵香气，营造出一种轻松而愉快的氛围。"
#negative_prompt     = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
prompt              = ""
negative_prompt     = ""

# Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability
# Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
# prompt                  = "A young woman with beautiful, clear eyes and blonde hair stands in the forest, wearing a white dress and a crown. Her expression is serene, reminiscent of a movie star, with fair and youthful skin. Her brown long hair flows in the wind. The video quality is very high, with a clear view. High quality, masterpiece, best quality, high resolution, ultra-fine, fantastical."
# negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
guidance_scale          = 6.0
seed                    = 43
num_inference_steps     = 50
lora_weight             = 0.55

# 初始化模型和管道
device = set_multi_gpus_devices(ulysses_degree, ring_degree)
config = OmegaConf.load(config_path)

transformer = WanTransformer3DModel.from_pretrained(
    "geoworld_stage2/transformer",
    transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)

if transformer_path is not None:
    print(f"From checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Vae
vae = AutoencoderKLWan.from_pretrained(
    os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
    additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
).to(weight_dtype)

if vae_path is not None:
    print(f"From checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file, safe_open
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

# Get Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
)

# Get Text encoder
text_encoder = WanT5EncoderModel.from_pretrained(
    os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
    additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
)
text_encoder = text_encoder.eval()

# Get Clip Image Encoder # vggt embedding
class VGGTWithDtype(VGGT):
    @property
    def dtype(self):
        return next(self.parameters()).dtype
vggt = VGGTWithDtype.from_pretrained("path to /vggt/ckpt").to(weight_dtype) 
#for param in clip_image_encoder.parameters():
#    param.requires_grad = False
vggt = vggt.eval()

clip_image_encoder = CLIPModel.from_pretrained(
    os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
).to(weight_dtype)
clip_image_encoder = clip_image_encoder.eval()

# Get Scheduler
Choosen_Scheduler = scheduler_dict = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]
if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
    config['scheduler_kwargs']['shift'] = 1
scheduler = Choosen_Scheduler(
    **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
)

# Get Pipeline
pipeline = WanFunControlPipeline(
    transformer=transformer,
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=scheduler,
    clip_image_encoder=clip_image_encoder,
    vggt=vggt,
).to("cuda") # vggt embedding
if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial
    transformer.enable_multi_gpus_inference()
    if fsdp_dit:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Add FSDP DIT")
    if fsdp_text_encoder:
        shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
        pipeline.text_encoder = shard_fn(pipeline.text_encoder)
        print("Add FSDP TEXT ENCODER")

if compile_dit:
    for i in range(len(pipeline.transformer.blocks)):
        pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
    print("Add Compile")

if GPU_memory_mode == "sequential_cpu_offload":
    replace_parameters_by_name(transformer, ["modulation",], device=device)
    transformer.freqs = transformer.freqs.to(device=device)
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(transformer, exclude_module_name=["modulation",], device=device)
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    #pass
    pipeline.to(device=device)

coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
if coefficients is not None:
    print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
    pipeline.transformer.enable_teacache(
        coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
    )

if cfg_skip_ratio is not None:
    print(f"Enable cfg_skip_ratio {cfg_skip_ratio}.")
    pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, num_inference_steps)

# 批处理逻辑
def process_folder(folder_path, output_dir, stage1_dir):
    """处理单个测试文件夹"""
    # 修复1: 使用正确的路径拼接方式
    mp4_path = os.path.join(stage1_dir, "output.mp4")
    png_path = os.path.join(folder_path, "1.png")
    
    # 修复2: 添加详细的文件存在检查
    print(f"Checking files in: {folder_path}")
    print(f"  MP4 path: {mp4_path} - {'Exists' if os.path.exists(mp4_path) else 'Missing'}")
    print(f"  PNG path: {png_path} - {'Exists' if os.path.exists(png_path) else 'Missing'}")
    
    # 验证文件存在
    if not os.path.exists(mp4_path) or not os.path.exists(png_path):
        print(f"  ! Missing files in {folder_path}, skipping")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输入路径
    control_video = mp4_path
    start_image = png_path
    ref_image = png_path
    
    print(f"Processing: {os.path.basename(folder_path)}")
    print(f"  Input video: {control_video}")
    print(f"  Input image: {start_image}")
    print(f"  Output dir: {output_dir}")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    with torch.no_grad():
        video_length_val = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
        latent_frames = (video_length_val - 1) // vae.config.temporal_compression_ratio + 1

        if enable_riflex:
            pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
        
        if ref_image is not None:
            clip_image = Image.open(ref_image).convert("RGB")
        elif start_image is not None:
            clip_image = Image.open(start_image).convert("RGB")
        else:
            clip_image = None
        
        if ref_image is not None:
            ref_image_latent = get_image_latent(ref_image, sample_size=sample_size)
        else:
            ref_image_latent = None
        
        if start_image is not None:
            start_image_latent = get_image_latent(start_image, sample_size=sample_size)
        else:
            start_image_latent = None

        #vggt full video
        vggt_video = []
        # Open video file
        cap = cv2.VideoCapture(control_video)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {control_video}")
        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.convert("RGB")
            vggt_video.append(img)
            # If frame is not read correctly, or we've reached the end of the video
            if not ret:
                break

        # 修复3: 使用局部变量避免名称冲突
        control_camera_txt_local = None
        
        if control_camera_txt_local is not None:
            input_video, input_video_mask = None, None
            control_camera_video = process_pose_file(control_camera_txt_local, sample_size[1], sample_size[0])
            control_camera_video = control_camera_video[:video_length_val].permute([3, 0, 1, 2]).unsqueeze(0)
        else:
            input_video, input_video_mask, _, _ = get_video_to_video_latent(
                control_video, video_length=video_length_val, sample_size=sample_size, fps=fps, ref_image=None
            )
            control_camera_video = None
        #print(input_video.size())

        sample = pipeline(
            prompt, 
            num_frames = video_length_val,
            negative_prompt = negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,

            control_video = input_video,
            control_camera_video = control_camera_video,
            ref_image = ref_image_latent,
            start_image = start_image_latent,
            clip_image = clip_image,
            shift = shift,
            vggt_video = vggt_video,
        ).videos

    
    # 保存结果
    save_video(sample, output_dir)
    return True

def save_video(sample, output_dir):
    """保存生成的视频"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, "output.mp4")
    
    # 确保样本数据在CPU上
    sample = sample.cpu()
    
    save_videos_grid(sample, video_path, fps=fps)
    print(f"  Video saved to: {video_path}")

# 主批处理循环
def batch_process():
    """处理测试集中的所有文件夹"""
    # 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(testset_dir) if f.is_dir()]
    
    print(f"Starting batch processing of {len(subfolders)} folders")
    print(f"Testset directory: {testset_dir}")
    print(f"Output base path: {base_save_path}")
    
    processed_count = 0
    skipped_count = 0
    
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        output_dir = os.path.join(base_save_path, folder_name)
        stage1_dir = os.path.join(base_stage1_path, folder_name)
        
        # 如果输出目录已存在且包含视频，跳过处理
        output_file = os.path.join(output_dir, "output.mp4")
        if os.path.exists(output_file):
            print(f"Skipping already processed folder: {folder_name}")
            skipped_count += 1
            continue
        
        # 处理当前文件夹
        success = process_folder(folder, output_dir, stage1_dir)
        
        if success:
            processed_count += 1
        else:
            skipped_count += 1
    
    print(f"\nProcessing summary:")
    print(f"  Total folders: {len(subfolders)}")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")

# 执行批处理
if __name__ == "__main__":
    batch_process()
    print("\nBatch processing completed!")