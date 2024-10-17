#! /bin/bash
if [ ${1} == "train" ]; then
    echo 'Training...'
    python segmentation_train.py --data_name KVASIR --data_dir ../Dataset/CVC/Train --out_dir ../Result/CVC_RES/Weight_V1 --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 16 --multi_gpu 0,1 --version 'v1' --input_channels_dist 4
fi
# if [ ${1} == "train" ]; then
#     echo 'Training...'
#     python segmentation_train.py --data_name ISIC --data_dir ../Dataset/ISIC_2018 --out_dir ../Result/ISIC_2018_RES/Weight_DWT_V4 --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 16 --multi_gpu 0,1 --version 'v1' --input_channels_dist 4
# fi
# if [ ${1} == "train" ]; then
#     echo 'Training...'
#     python segmentation_train.py --data_name ISIC --data_dir ../Dataset/ISIC_2018 --out_dir ../Result/ISIC_2018_RES/Weight_DWT_V4 --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 16 --multi_gpu 0,1 --version 'v1' --input_channels_dist 4
# fi
# if [ ${1} == "sample" ]; then
#     echo 'Sampling...'
#     python segmentation_sample.py --data_name BRATS --data_dir ../Dataset/Processed_BraTs2021/testing --out_dir ../Result/BRATS_RES/Images_DWT_V7 --model_path ../Result/BRATS_RES/Weight_DWT_V5/emasavedmodel_0.9999_100000.pt --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5 --version 'v1' --diffusion_steps 50 --dpm_solver True
if [ ${1} == "sample" ]; then
    echo 'Sampling...'
    python segmentation_sample.py --data_name KVASIR --data_dir ../Dataset/CVC/CVC-300 --out_dir ../Result/CVC_RES/CVC-300_V2 --model_path ../Result/CVC_RES/Weight_V1/emasavedmodel_0.9999_090000.pt --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5 --version 'v1' --diffusion_steps 50 --dpm_solver True
fi

if [ ${1} == "evl" ]; then
    echo 'Computing...'
    python scripts/segmentation_env.py --inp_pth Result/ISIC_RES/Images_DWT_V8 --out_pth Dataset/ISIC_DATA/ISBI2016_ISIC_Part1_Test_GroundTruth

fi
