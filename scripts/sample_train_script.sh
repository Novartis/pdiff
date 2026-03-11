#!/bin/bash

#$ -l m_mem_free=40G
#$ -l h_rt=6:00:00
#$ -l gpu_card=1
#$ -V
#$ -cwd
#$ -R y

#cd $pdiff_dir
# TODO set CUDA_VISIBLE_DEVICES and other config as needed


export experiment_name="sample_training"
# could also download the stable-diffusion 2.1
#export model_path="stabilityai/stable-diffusion-2-1"
export model_path="../tests/test_data/sample_pdiff_model"
export output_dir="../models/"$experiment_name
export logging_dir="../logs/"
# for demonstration purposes, we'll use the same dataset for training, validation, and visual inference. Change these as appropriate
export training_dataset_file_path="../tests/test_data/prepared_metadata.pkl"
export validation_dataset_file_path="../tests/test_data/prepared_metadata.pkl"
export visual_inference_dataset_file_path="../tests/test_data/prepared_metadata.pkl"

accelerate launch --config_file ./accelerate_1gpu_fp32.yaml --main_process_port 15001 --gpu_ids $CUDA_VISIBLE_DEVICES ../src/pdiff/training.py \
  --from_scratch \
  --pretrained_model_name_or_path=$model_path \
  --training_dataset_file_path=$training_dataset_file_path \
  --validation_dataset_file_path=$validation_dataset_file_path \
  --visual_inference_dataset_file_path=$training_dataset_file_path \
  --logging_dir=$logging_dir \
  --tracker_project_name=$experiment_name \
  --resolution=512  \
  --train_batch_size=4 \
  --checkpoints_total_limit=5 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=1000 \
  --output_dir=$output_dir \
  --report_to="tensorboard" \
  --num_train_epochs=1000 \
  --checkpointing_steps=1000 \
  --validation_epochs=100 \
  --max_grad_norm=.1 \
  --dataloader_num_workers=4 \
  --resume_from_checkpoint "latest" \
  --seed=42 \
  --noise_offset=0.2 \
  --snr_gamma=5.0 \
  --input_perturbation=0.1 \
