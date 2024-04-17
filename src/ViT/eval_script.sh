python run_mae.py \
    --dataset_name nateraw/image-folder \
    --train_dir "/home/jackmorris/football/football2text/data/jpeg_data/train/*" \
    --image_column_name image \
    --output_dir /home/jackmorris/football/football2text/models/ViT/2 \
    --remove_unused_columns False \
    --label_names pixel_values \
    --mask_ratio 0.75 \
    --do_eval \
    --per_device_eval_batch_size 200 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 150 \
    --save_strategy eval \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \