import torch
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
import numpy.random as rand
# model = VisionTextDualEncoderModel.from_vision_text_pretrained(
#     "openai/clip-vit-base-patch32", "jkruk/distilroberta-base-ft-nfl"
# )

# tokenizer = AutoTokenizer.from_pretrained("jkruk/distilroberta-base-ft-nfl")
# image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
# processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# model = RobertaForMaskedLM.from_pretrained('jkruk/distilroberta-base-ft-nfl')
# print(image_processor(train_data[0][1]))
model = RobertaForMaskedLM.from_pretrained("/media/test-mlm")
tokenizer = AutoTokenizer.from_pretrained("/media/test-mlm")

if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parents[1].parent / "data"
    # print(data_dir)
    which = 'train'
    ds = NFLTextDataset(data_dir,which="test")
    randidx = rand.randint(0,2000)
    # ds.build_csvs()
    # print(ds[randidx])
    # exit()
    # q = input("Input query here. Mask predicted word with <mask>:\n")
    # q = "Tom Brady passed the <mask> to his teammate."
    q = ds[randidx]

    #masking random word
    qmask = q.split()
    mask_index = rand.randint(3, len(qmask) - 1)
    qmask[mask_index] = '<mask>'
    qmask = ' '.join(qmask)

    inputs = tokenizer(qmask,return_tensors='pt')
    outputs = model.forward(**inputs)
    predictions = outputs.logits
    predicted_indices = torch.softmax(predictions[0, inputs['input_ids'][0] == tokenizer.mask_token_id], dim=-1)
    
    top_probs, top_indices = predicted_indices.topk(k=5)
    print(q)
    print(qmask)
    for idx,prob in zip(top_indices[0],top_probs[0]):
        predicted_token = tokenizer.decode(idx)
        print(predicted_token,round((float(prob)),3))
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