# -*- coding: utf-8 -*-
"""Project-NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_lsro-IWPyoDBi9wwRMTg8BdW14nnBeM
"""

!nvidia-smi #nvidia gpu information

from google.colab import drive
drive.mount('/content/drive')

!lscpu #cpu information

pip install datasets

pip install transformers

pip install --upgrade accelerate

pip install --upgrade torch

pip install --upgrade transformers

pip install streamlit

import torch
import os
import time

import pandas as pd
from datasets import load_metric
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import (
    DistilBertConfig,
    AutoConfig,
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    default_data_collator
)
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import logging
logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device('cuda') #NVIDIA-GPU

"""The code is utilizing an NVIDIA GPU for accelerated computation.

### **Data Loading & Preprocessing**
"""

Humord_df= pd.read_csv('/content/drive/MyDrive/dataset.csv')
Humord_df

Humord_df["label"] = Humord_df["humor"].astype(int) #convert True/False into a 0 or a 1
Humord_df

train, test = train_test_split(Humord_df, test_size=0.2, random_state=42)

len(train)

len(test)

DistilBertConfig()

"""It is used to define the configuration or settings for the DistilBERT model.

### **Tokenizing the text into integer vectors**
"""

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenizer

"""Here, I create an instance of the tokenizer using DistilBertTokenizerFast. The tokenizer has specific configurations that we can observe. The vocabulary size is 200K, and when tokenizing text, padding is applied to the right side of the tokenized vector. Additionally, there are special tokens such as SEP (separator token) and PAD (pad token) that serve specific purposes."""

train_encodings = tokenizer(list(train['text']), padding='max_length', truncation=True, max_length=50)
test_encodings = tokenizer(list(test['text']), padding='max_length', truncation=True, max_length=50)

"""Preprocessing the text data for both the training and testing datasets, To ensure consistency across all datasets, the provided code pads the integer sequences to a specific length (in this case, 50). This is done to maintain a uniform input size for the model. By having vectors of the same length, we can meet the model's expectation and ensure that the input size remains consistent."""

test_encodings.keys()

"""Once tokenized, we can see that the data have two keys: input_ids and attention_mask, These input IDs and attention masks are essential for feeding the data into the BERT model for training or inference, as they represent the tokenized input sequences and provide information on which tokens to attend to during processing."""

print(test_encodings['input_ids'])

"""### **Analyzing the text**"""

#all_nonhumorous_words = ''
#for idx,text_data in enumerate(list(train['text'])):
#    if list(train['label'])[idx] == 0:
#        all_nonhumorous_words += ' ' + text_data.strip()


#wc = WordCloud(width=1024,height=1024, min_font_size=8, stopwords=STOPWORDS).generate(all_nonhumorous_words)

"""Analysis non-humorous words - We will use a word cloud to visualize the most common and least common words. We will use training data"""

#all_nonhumorous_words[0:500]

#len(all_nonhumorous_words)

#plt.figure(figsize = (8, 8), facecolor = None)
#plt.imshow(wc)
#plt.axis("off")
#plt.tight_layout(pad = 0)

#plt.show()

#all_humorous_words = ''
#for idx,text_data in enumerate(list(train['text'])):
#   if list(train['label'])[idx] == 1:
#       all_humorous_words += ' ' + text_data.strip()

#wc2 = WordCloud(width=1024,height=1024, min_font_size=8, stopwords=STOPWORDS).generate(all_humorous_words)

#plt.figure(figsize = (8, 8), facecolor = None)
#plt.imshow(wc2)
#plt.axis("off")
#plt.tight_layout(pad = 0)

#plt.show()

"""### **Model Training**"""

class newDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

"""To utilize the computational power and benefits of PyTorch, it is important to convert the tokenized data into PyTorch tensors. PyTorch tensors allow for efficient memory usage, parallel processing, and seamless integration with the PyTorch library and models."""

train_dataset = newDataset(train_encodings, list(train['label']))
test_dataset = newDataset(test_encodings, list(test['label']))

"""1. newDataset(train_encodings, list(train['label'])): This line instantiates the newDataset class and passes two arguments: train_encodings and list(train['label']).

2. train_encodings: This argument contains the tokenized and preprocessed input text data of the training dataset. It typically includes the encoded sequences, attention masks, and other necessary information generated by the tokenizer.

3. list(train['label']): This argument contains the corresponding labels for the training data. It is a list of labels that indicate whether each example is humorous or non-humorous. The length of this list should match the number of instances in the training dataset.
"""

device = torch.device('cuda')

index = torch.LongTensor(test_dataset[0]['input_ids']).to(device).unsqueeze(0)
attn_mask =  torch.LongTensor(test_dataset[0]['attention_mask']).to(device).unsqueeze(0)
print(f"Original sentence = {list(test['text'])[0]}")
print(f"Original label = {list(test['label'])[0]}.")
print(f"Original humor = {list(test['humor'])[0]}.")
print(f'index={index}')
print(f'attn_mask={attn_mask}')

"""Now that we have finished preparing the data, let's compare an example from the original dataset with its preprocessed version for training."""

decoded = tokenizer.decode(test_dataset[0]['input_ids'])
decoded

"""here, we decoded the tokens and we can se that there are sep and pad......"""

list(test['label'])[0] #0 = nonhumorous, 1=humorous

"""checking the test dataset of '0' th place"""

test_dataset[0]

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive",
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    logging_steps=350,
    report_to="none"
)

"""Here we are using 'GPU'

I have provided the number of training epochs, the batch size, the number of text examples to handle at once in memory on the GPU, and the config from the **Transformer - TrainingArguments** framework.
"""

metric = load_metric("f1") #can also put "accuracy" if desired.
def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

"""Validation Rules -
Currently creating a metrics function to evaluate the performance of the model during training. This function will use the **Test** dataset to measure the progress of the model at specific intervals. In this case, I am using the F1 metric, which is a commonly used evaluation metric in classification tasks.
"""

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.to(device)
print('')

"""loaded the pretrained model, and make sure to place the model on the GPU device."""

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset            # evaluation dataset
    compute_metrics=compute_metrics
    )

start_time = time.time()
#start training
trainer.train()
trainer.save_model('/content/drive/MyDrive')
train_time = time.time()-start_time
print(f'train_time={train_time}')

"""Next, I create an instance of the trainer using the Trainer class. I provide the trainer with the training arguments (`training_args`) that were defined earlier, as well as the training dataset and the test dataset. Additionally, I pass the `compute_metrics` function that I defined, which will be used to compute evaluation metrics during training.


By observing the training loss, we can notice that it consistently decreases as the training progresses, which is a desirable outcome. You have the flexibility to fine-tune various training parameters to optimize the performance of your model. Additionally, the model will be saved at multiple checkpoints, which can be found in the "output_results" folder.

### **Evaluation**
"""

#model loading
output_model_folder = '/content/drive/MyDrive' #this may change depending on where you saved your model.
Model_S = DistilBertForSequenceClassification.from_pretrained(output_model_folder)
Model_S.to(device)
print('')

"""Saving the model for evaluation purpose, now we are testing the test dataset"""

class newDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

trainer = Trainer(
    model=Model_S,                    # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,
    eval_dataset=test_dataset,        # training dataset         # evaluation dataset
    compute_metrics=compute_metrics
    )

print("**************** Evaluation below************")
metrics = trainer.evaluate()
metrics["eval_samples"] = len(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

"""## **F1 score is 98% which is really amazing!!!!**"""

index = torch.LongTensor(test_dataset[0]['input_ids']).to(device).unsqueeze(0)
attn_mask =  torch.LongTensor(test_dataset[0]['attention_mask']).to(device).unsqueeze(0)
print(f'index={index}')
print(f'attn_mask={attn_mask}')

pred = Model_S(index)
pred

torch.argmax(pred.logits)

"""### **Inference on single sample**"""

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("/content/drive/MyDrive")

def predict_humor(statement):
    encoded_input = tokenizer.encode_plus(
        statement,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    logits = model(**encoded_input).logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label

statement = "I am cat"
predicted_label = predict_humor(statement)
if predicted_label == 1:
    print("The statement is humorous.")
else:
    print("The statement is not humorous.")

"""# I've taken Friends serial dialogues and calculated % of humor."""

import pandas as pd

# Assuming the dialogues are stored in a CSV file named "dialogues.csv"
dialogues_df = pd.read_csv("/content/drive/MyDrive/Revised FRIENDS dataset.csv")

dialogues = dialogues_df["dialogue"].tolist()

# Tokenize the dialogues using the BERT tokenizer
encoded_inputs = tokenizer(dialogues, padding="longest", truncation=True, return_tensors="pt")

input_ids = encoded_inputs.input_ids.to(device)
attention_mask = encoded_inputs.attention_mask.to(device)

with torch.no_grad():
    logits = Model_S(input_ids, attention_mask=attention_mask).logits

# Obtain the predicted labels by selecting the class with the highest probability
predictions = torch.argmax(logits, dim=-1).tolist()

for i in range(len(dialogues)):
    dialogue = dialogues[i]
    prediction = predictions[i]
    label = "Humorous" if prediction == 1 else "Non-humorous"
    print(f"Dialogue: {dialogue}\nPrediction: {label}\n")

predictions_df = pd.DataFrame({"Dialogue": dialogues, "Prediction": predictions})
predictions_df["Label"] = predictions_df["Prediction"].map({1: "Humorous", 0: "Non-humorous"})
print(predictions_df)

predictions_df.to_excel("FRIENDS_predictions.xlsx", index=False)

humor_count = sum(predictions)
total_count = len(predictions)
non_humor_count = total_count - humor_count

humor_percentage = (humor_count / total_count) * 100
non_humor_percentage = (non_humor_count / total_count) * 100

print(f"Humor Percentage: {humor_percentage}%")
print(f"Non-Humor Percentage: {non_humor_percentage}%")