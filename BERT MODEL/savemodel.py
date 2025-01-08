from transformers import BertForSequenceClassification
import torch

def save_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.save_pretrained('./sentiment_model')
    tokenizer.save_pretrained('./sentiment_model')

save_model()
