import torch.cuda
import argparse
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
import os
from torch.nn.functional import interpolate

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda'
    LLaVA_device = 'cuda'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# hyparams here
parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", type=str)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--min_size", type=int, default=1024)
parser.add_argument("--edm_steps", type=int, default=50)
parser.add_argument("--s_stage1", type=int, default=-1)
parser.add_argument("--s_churn", type=int, default=5)
parser.add_argument("--s_noise", type=float, default=1.01)
parser.add_argument("--s_cfg", type=float, default=4.0)
parser.add_argument("--s_stage2", type=float, default=1.)
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--a_prompt", type=str,
                    default='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                            'hyper sharpness, perfect without deformations.')
parser.add_argument("--n_prompt", type=str,
                    default='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                            'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                            'deformed, lowres, over-smooth')
parser.add_argument("--color_fix_type", type=str, default='Wavelet', choices=["None", "AdaIn", "Wavelet"])
parser.add_argument("--linear_CFG", action='store_true', default=True)
parser.add_argument("--linear_s_stage2", action='store_true', default=False)
parser.add_argument("--spt_linear_CFG", type=float, default=1.0)
parser.add_argument("--spt_linear_s_stage2", type=float, default=0.)
parser.add_argument("--ae_dtype", type=str, default="bf16", choices=['fp32', 'bf16'])
parser.add_argument("--diff_dtype", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--loading_half_params", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
args = parser.parse_args()
print(args)
use_llava = not args.no_llava

# load SUPIR
model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign=args.SUPIR_sign)
if args.loading_half_params:
    model = model.half()
if args.use_tile_vae:
    model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
model.ae_dtype = convert_dtype(args.ae_dtype)
model.model.dtype = convert_dtype(args.diff_dtype)
model = model.to(SUPIR_device)
# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None

os.makedirs(args.save_dir, exist_ok=True)
DEBUG = True

print("-"*60)
print("Start upscale image!")

excluded_dir = [
    "1_lam_thao_anh",
    "3_nguyen_huy_hoang",
    "4_nguyen_phuong_trang",
    "5_pham_thi_phuong_lan",
    "6_vu_quoc_hoang",
    "7_nguyen_huy_manh",
    "8_nguyen_huy_manh_1",
    "9_nguyen_ha_linh",
    "11_phung_trong_hieu",
    "12_pham_huong_giang",
    "14_nguyen_thi_kim_cuc",
    "16_ha_thi_linh_chi",
    # "17_nguyen_huy_manh_3",
    "19_bui_son_anh",
    "20_vu_quynh_huong",
    "21_luu_trong_duy",
    "23_trieu_thanh_tung",
    "25_do_quynh_thu",
    "27_tran_minh_anh",
    "30_tomoe_shinagawa",
    "34_doan_the_vinh",
    "35_miu",
    "38_thai_cam_tu",
    "39_trung_thanh_tran",
    "40_mai_nhat_anh",
    "41_dao_bao_ngoc",
    "42_do_tien_dinh",
    "43_gia_huy",
    "44_hong_ba",
    "45_nguyen_quynh_huong",
    "46_pham_thuy_linh",
]
for root, dirs, files in os.walk(args.img_dir):

    for file_name in files:
        if not file_name.lower().endswith((".jpeg", ".jpg", ".png", ".bmp", ".tiff", ".webp")):
            continue
    
        img_path = os.path.join(root, file_name)
        img_name = os.path.splitext(file_name)[0]

        rel_path = os.path.relpath(root, args.img_dir)

        # if rel_path in excluded_dir:
        #     continue
        
        print(f"Upscaling {rel_path}")
        save_subdir = os.path.join(args.save_dir, rel_path)
        
        os.makedirs(save_subdir, exist_ok=True)
    
        LQ_ips = Image.open(img_path)
        width, height = LQ_ips.size
        
        # avoid out of memory error
        if width >= 1000 or height >= 1000:
            upscale = 1
        elif width >= 500 or height >= 500:
            upscale = 2
        elif width >= 300 or height >= 300:
            upscale = 4
        else:
            upscale = 8
            
        LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=upscale, min_size=args.min_size)

        LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

        # step 1: Pre-denoise for LLaVA, resize to 512
        LQ_img_512, h1, w1 = PIL2Tensor(LQ_ips, upsacle=upscale, min_size=args.min_size, fix_resize=512)
        LQ_img_512 = LQ_img_512.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
        clean_imgs = model.batchify_denoise(LQ_img_512)
        clean_PIL_img = Tensor2PIL(clean_imgs[0], h1, w1)

        # step 2: LLaVA
        if use_llava:
            captions = llava_agent.gen_image_caption([clean_PIL_img])
        else:
            captions = ['']
        print(captions)

        # step 3: Diffusion Process
        samples = model.batchify_sample(LQ_img, captions, num_steps=args.edm_steps, restoration_scale=args.s_stage1, s_churn=args.s_churn,
                                        s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2, seed=args.seed,
                                        num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt, color_fix_type=args.color_fix_type,
                                        use_linear_CFG=args.linear_CFG, use_linear_control_scale=args.linear_s_stage2,
                                        cfg_scale_start=args.spt_linear_CFG, control_scale_start=args.spt_linear_s_stage2)
        
        # save
        for _i, sample in enumerate(samples):
            output_path = os.path.join(save_subdir, f"{img_name}_{_i}.png")
            Tensor2PIL(sample, h0, w0).save(output_path)
        
        torch.cuda.empty_cache()

print("-"*60)
print("Done upscale image!")

# for img_pth in os.listdir(args.img_dir):
#     img_name = os.path.splitext(img_pth)[0]
    
    
#     LQ_ips = Image.open(os.path.join(args.img_dir, img_pth))
#     LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=args.upscale, min_size=args.min_size)
#     LQ_img = LQ_img.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]

#     # step 1: Pre-denoise for LLaVA, resize to 512
#     LQ_img_512, h1, w1 = PIL2Tensor(LQ_ips, upsacle=args.upscale, min_size=args.min_size, fix_resize=512)
#     LQ_img_512 = LQ_img_512.unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
#     clean_imgs = model.batchify_denoise(LQ_img_512)
#     clean_PIL_img = Tensor2PIL(clean_imgs[0], h1, w1)

#     # step 2: LLaVA
#     if use_llava:
#         captions = llava_agent.gen_image_caption([clean_PIL_img])
#     else:
#         captions = ['']
#     print(captions)

#     # # step 3: Diffusion Process
#     samples = model.batchify_sample(LQ_img, captions, num_steps=args.edm_steps, restoration_scale=args.s_stage1, s_churn=args.s_churn,
#                                     s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2, seed=args.seed,
#                                     num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt, color_fix_type=args.color_fix_type,
#                                     use_linear_CFG=args.linear_CFG, use_linear_control_scale=args.linear_s_stage2,
#                                     cfg_scale_start=args.spt_linear_CFG, control_scale_start=args.spt_linear_s_stage2)
#     # save
#     for _i, sample in enumerate(samples):
#         Tensor2PIL(sample, h0, w0).save(f'{args.save_dir}/{img_name}_{_i}.png')

