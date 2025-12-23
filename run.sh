export CUDA_VISIBLE_DEVICES=0

python -m test --img_dir /home/trangnguyenphuong/SUPIR/output/processed_data_7b3aeaf1d9d44c568a8a03d603c5b008_upscaled_refine \
                --metadata /home/trangnguyenphuong/SUPIR/output/upscaled_images.csv \
                --save_dir /home/trangnguyenphuong/SUPIR/output/test_final \
                --no_llava --loading_half_params --use_tile_vae
