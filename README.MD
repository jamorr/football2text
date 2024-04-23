# Football2Text
Deep learning project for turning American football tracking data into text.

[NFL Big Data Bowl 2024 dataset](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/overview)

# Proposed Framework
Image2Text
- Place all data from an NFL Big Databowl on kaggle in the data directory
- Run the below command to create all jpegs/mp4s from the tracking data as well as split the data into train/test/evaluation sets
- `python main.py --prep-data --data-path <path to your data directory>`
- Now all data should be organized into parquet files which are parsable by the huggingface dataset module
- To use our scripts it is recommended that you move the README.md file from the /assets directory to the /data/CLIP directory.
- The directory structure should be as follows:
    ├─/data
    │ ├─/CLIP
    │ │ ├─train
    │ │ ├─test
    │ │ ├─validation
    │ │ └─README.md
    │ ├─/train
    │ ├─/test
    │ ├─/val
    │ └─<csv and parquet files>
    ├─/models
    ├─/assets
    ├─/src
    ├─README.MD
    └─requirements.yml

- To train the model each individual piece must first be trained in sequence
    1. Train RoBERTa and ViT MAE on the text and image data respectively
        a. RoBERTa may be fine tuned, but ViT MAE will need to be trained from scratch
    2. Train a CLIP model using RoBERTa and the ViT encoder + an untrained classification head
    3. Train GPT2 using the pretrained CLIP model to generate image embeddings as prefixes for the model
- The resulting model now should take in image data and produce a caption for it

# Requirements
See environment.yml for requirements
