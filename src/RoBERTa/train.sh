#Training RoBERTa on ouput
python run_mlm.py \
    --model_name_or_path jkruk/distilroberta-base-ft-nfl \
    --train_file /home/jaydenfassett/football2text/src/roberta/data/roberta2/train.csv \
    --validation_file /home/jaydenfassett/football2text/src/roberta/data/roberta2/test.csv \
    --per_device_train_batch_size 25 \
    --per_device_eval_batch_size 25 \
    --num_train_epochs 40 \
    --do_train \
    --do_eval \
    --do_predict \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --output_dir /media/jj_data/models/roberta3/ \
    --overwrite_output_dir yes \