python og_train.py \
    --dataset_name /media/jj_data/data/CLIP \
    --dataset_config_name image_only \
    --image_column_name pixel_values \
    --output_dir /media/jj_data/models/ViT/fixed_params \
    --mask_ratio 0.50 \
    --norm_pix_loss \
    --do_train \
    --do_eval \
    --remove_unused_columns False \
    --base_learning_rate 1.5e-4 \
    --lr_scheduler_type cosine \
    --weight_decay 0.05 \
    --num_train_epochs 15 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 150 \
    --per_device_eval_batch_size 50 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --metric_for_best_model eval_loss \
    --save_strategy best \
    --save_total_limit 3 \
    --dataloader_num_workers 4 \
    --seed 1337 \