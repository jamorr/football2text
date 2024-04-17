import pathlib
from transformers import ViTMAEConfig, ViTMAEForPreTraining, TrainingArguments, Trainer, logging
from dataset import NFLImageDataset
logging.set_verbosity_error()


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Initializing a ViT MAE vit-mae-base style configuration
configuration = ViTMAEConfig()

# Initializing a model (with random weights) from the vit-mae-base style configuration
model = ViTMAEForPreTraining(configuration)

# Accessing the model configuration
# default_args = {
#     "output_dir": "tmp",
#     "evaluation_strategy": "steps",
#     "num_train_epochs": 1,
#     "log_level": "error",
#     "report_to": "none",
# }
repo_dir = pathlib.Path(__file__).parents[2]
training_args = TrainingArguments(output_dir=str(repo_dir/"models"), per_gpu_train_batch_size=40, do_train=True)
data_dir = repo_dir/"data"/"train"
dataset = NFLImageDataset(data_dir)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

result = trainer.train()

# print_summary(result)

