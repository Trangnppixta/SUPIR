## 🔧 Dependencies and Installation

1. Clone repo
    ```bash
    git clone https://github.com/Fanghua-Yu/SUPIR.git
    cd SUPIR
    ```

2. Install dependent packages
    ```bash
    conda create -n SUPIR python=3.8 -y
    conda activate SUPIR
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

3. Download Checkpoints

For users who can connect to huggingface, please setting `LLAVA_CLIP_PATH, SDXL_CLIP1_PATH, SDXL_CLIP2_CKPT_PTH` in `CKPT_PTH.py` as `None`. These CLIPs will be downloaded automatically. 

#### Dependent Models
* [SDXL CLIP Encoder-1](https://huggingface.co/openai/clip-vit-large-patch14)
* [SDXL CLIP Encoder-2](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
* [SDXL base 1.0_0.9vae](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors)
* [LLaVA CLIP](https://huggingface.co/openai/clip-vit-large-patch14-336)
* [LLaVA v1.5 13B](https://huggingface.co/liuhaotian/llava-v1.5-13b)


#### Models we provided:
* `SUPIR-v0Q`: [Baidu Netdisk](https://pan.baidu.com/s/1lnefCZhBTeDWijqbj1jIyw?pwd=pjq6), [Google Drive](https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR?usp=sharing)
    
    Default training settings with paper. High generalization and high image quality in most cases.

* `SUPIR-v0F`: [Baidu Netdisk](https://pan.baidu.com/s/1AECN8NjiVuE3hvO8o-Ua6A?pwd=k2uz), [Google Drive](https://drive.google.com/drive/folders/1yELzm5SvAi9e7kPcO_jPp2XkTs4vK6aR?usp=sharing)

    Training with light degradation settings. Stage1 encoder of `SUPIR-v0F` remains more details when facing light degradations.

4. After downloading above checkpoints, replace those params in SUPIR/options/SUPIR_v0.yaml with path/to/your/ckpt
    ```
    SDXL_CKPT: /home/trangnguyenphuong/SUPIR/sd_xl_base_1.0_0.9vae.safetensors
    SUPIR_CKPT_F: /home/trangnguyenphuong/SUPIR/SUPIR-v0F.ckpt
    SUPIR_CKPT_Q: /home/trangnguyenphuong/SUPIR/SUPIR-v0Q.ckpt
    ```
---

## ⚡ Quick Inference
### Python Script
Perform super-resolution on images for ai-headshot on RTX 3090 24GB VRAM:
```Shell
# Only loading half-params and not using llava to avoid out-of-memory
bash run.sh
```
### Usage of SUPIR
```Shell
Usage: 
-- python test.py [options] 
-- python gradio_demo.py [interactive options]

--img_dir                Input folder.
--save_dir               Output folder.
--upscale                Upsampling ratio of given inputs. Default: 1
--SUPIR_sign             Model selection. Default: 'Q'; Options: ['F', 'Q']
--seed                   Random seed. Default: 1234
--min_size               Minimum resolution of output images. Default: 1024
--edm_steps              Numb of steps for EDM Sampling Scheduler. Default: 50
--s_stage1               Control Strength of Stage1. Default: -1 (negative means invalid)
--s_churn                Original hy-param of EDM. Default: 5
--s_noise                Original hy-param of EDM. Default: 1.01
--s_cfg                  Classifier-free guidance scale for prompts. Default: 4.0
--s_stage2               Control Strength of Stage2. Default: 1.0
--num_samples            Number of samples for each input. Default: 1
--a_prompt               Additive positive prompt for all inputs. 
    Default: 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, 
    hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme
     meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'
--n_prompt               Fixed negative prompt for all inputs. 
    Default: 'painting, oil painting, illustration, drawing, art, sketch, oil painting, 
    cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, 
    low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'
--color_fix_type         Color Fixing Type. Default: 'Wavelet'; Options: ['None', 'AdaIn', 'Wavelet']
--linear_CFG             Linearly (with sigma) increase CFG from 'spt_linear_CFG' to s_cfg. Default: True
--linear_s_stage2        Linearly (with sigma) increase s_stage2 from 'spt_linear_s_stage2' to s_stage2. Default: False
--spt_linear_CFG         Start point of linearly increasing CFG. Default: 1.0
--spt_linear_s_stage2    Start point of linearly increasing s_stage2. Default: 0.0
--ae_dtype               Inference data type of AutoEncoder. Default: 'bf16'; Options: ['fp32', 'bf16']
--diff_dtype             Inference data type of Diffusion. Default: 'fp16'; Options: ['fp32', 'fp16', 'bf16']
```
