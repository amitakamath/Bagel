# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

GPUS=8
#model_path=./eval/gen/geneval/model/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth
model_path=/checkpoint/dream/transfusion/cache/BAGEL-7B-MoT
output_path=./outputs_with_thinking_short


# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp_with_thinking.py \
    --output_dir $output_path/images \
    --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \
    --batch_size 1 \
    --num_images 1 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \
    --think
    # --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \

# calculate score
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
    $output_path/images \
    --outfile $output_path/results.jsonl \
    --model-path ./eval/gen/geneval/model


# summarize score
python ./eval/gen/geneval/evaluation/summary_scores.py $output_path/results.jsonl
