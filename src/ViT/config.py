from transformers import ViTMAEConfig, ViTMAEModel, TrainingArguments, Trainer, logging
from src.dataset import NFLDataset
logging.set_verbosity_error()

# Initializing a ViT MAE vit-mae-base style configuration
configuration = ViTMAEConfig()

# Initializing a model (with random weights) from the vit-mae-base style configuration
model = ViTMAEModel(configuration)

# Accessing the model configuration
default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}
training_args = TrainingArguments(per_device_train_batch_size=40, **default_args)
dataset = NFLDataset()
trainer = Trainer(model=model, args=training_args, train_dataset=)

result = trainer.train()

print_summary(result)

