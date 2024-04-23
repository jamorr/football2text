---
version:
  - 0.0.1
language:
  - en
multilinguality:
  - monolingual
tags:
  - image-text pairs
task_ids:
  - image-captioning
task_categories:
  - text-to-image
  - image-to-text
configs:
  - config_name: main
  - config_name: image_only
  - config_name: text_only
dataset_info:
  config_name: main
  features:
    - name: pixel_values
      dtype: image
    - name: input_ids
      dtype: string
  config_name: image_only
  features:
    - name: pixel_values
      dtype: image
  config_name: text_only
  features:
    - name: input_ids
      dtype: string
---
