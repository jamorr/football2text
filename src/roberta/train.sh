#Training RoBERTa on ouput
python run_mlm.py \
    --model_name_or_path jkruk/distilroberta-base-ft-nfl \
    --train_file /home/jaydenfassett/football2text/src/roberta/data/roberta2/train.csv \
    --validation_file /home/jaydenfassett/football2text/src/roberta/data/roberta2/val.csv \
    --per_device_train_batch_size 25 \
    --per_device_eval_batch_size 25 \
    --num_train_epochs 40 \
    --do_train \
    --do_eval \
    --logging_steps 50 \
    --eval_steps 75 \
    --output_dir /media/jj_data/models/roberta/ \
    --overwrite_output_dir yes \