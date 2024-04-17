from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor,
    RobertaForMaskedLM
)
from dataset import NFLTextDataset
import pathlib
import numpy as np
import pandas as pd
# model = VisionTextDualEncoderModel.from_vision_text_pretrained(
#     "openai/clip-vit-base-patch32", "jkruk/distilroberta-base-ft-nfl"
# )

# tokenizer = AutoTokenizer.from_pretrained("jkruk/distilroberta-base-ft-nfl")
# image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
# processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

model = RobertaForMaskedLM.from_pretrained('jkruk/distilroberta-base-ft-nfl')
# print(image_processor(train_data[0][1]))



if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parents[1].parent / "data"
    # print(data_dir)
    which = 'train'
    ds = NFLTextDataset(data_dir,which)
    ds.build_csvs()
    inpt = ds[0]
    # print(model(**inpt))





# tks = tokenizer.tokenize(q)
# model.forward(tks)
# outputs = model(input_ids=processed_text.input_ids, 
#                 attention_mask=processed_text.attention_mask, 
#                 pixel_values=processed_image.pixel_values)
# print(model)

# save the model and processor
# model.save_pretrained("clip-roberta")
# processor.save_pretrained("clip-roberta")