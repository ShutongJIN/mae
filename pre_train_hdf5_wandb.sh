
module load Anaconda/2021.05-nsc1
conda activate data4robotics

python submitit_pretrain.py \
    --job_dir /proj/cloudrobotics-nest/users/Stacking/dataset/CloudGripper_push_1k/Ball/pre_trained_weights \
    --nodes 1 \
    --ngpus 8 \
    --partition berzelius \
    --batch_size 256 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /proj/cloudrobotics-nest/users/Stacking/dataset/CloudGripper_push_1k/Ball_Single_Top_hdf5