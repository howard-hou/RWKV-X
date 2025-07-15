export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_MODE=offline

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --load_model /root/autodl-tmp/RWKV-X-3.6B-64k-Base.pth \
    --wandb "rwkvx3b6-64k-ultrachat-sft" --proj_dir out/rwkvx3b6-64k-ultrachat-sft \
    --data_file /root/autodl-tmp/ultrachat_200k.jsonl \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 4096 --epoch_steps 1000 --epoch_count 3 --epoch_begin 0 --epoch_save 1 \
    --micro_bsz 16 --accumulate_grad_batches 2 --n_layer 32 --n_embd 2560 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 4 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --enable_progress_bar True
