---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
model-index:
- name: distilbert-base-uncased-finetuned-emotion
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-base-uncased-finetuned-emotion

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on SetFit/emotion dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2276
- Accuracy: 0.921
- F1: 0.9209

https://huggingface.co/goldenk/distilbert-base-uncased-finetuned-emotion

## Model description

This model follows chapter 2 of https://github.com/nlp-with-transformers/notebooks. A few things that were changed from the original notebook:

- the emotion dataset has moved to SetFit/emotion https://github.com/nlp-with-transformers/notebooks/issues/77
- the new dataset doesn't have ClassLabel feature so needed to change int2str method https://github.com/nlp-with-transformers/notebooks/issues/77
- made the label names on inference API human-readable with https://discuss.huggingface.co/t/change-label-names-on-inference-api/3063/3
- function to inspect dataset for existence of certain strings

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|
| 0.8732        | 1.0   | 250  | 0.3279          | 0.9055   | 0.9037 |
| 0.259         | 2.0   | 500  | 0.2276          | 0.921    | 0.9209 |


### Framework versions

- Transformers 4.13.0
- Pytorch 1.13.0+cu116
- Datasets 1.16.1
- Tokenizers 0.10.3
