from transformers import BertForSequenceClassification
from preprocess import preprocess_data
from transformers import Trainer
import torch

model = BertForSequenceClassification.from_pretrained('./sentiment_model')
train_data, test_data = preprocess_data()

trainer = Trainer(
    model=model,
    eval_dataset=test_data
)

results = trainer.evaluate()
print(f"Test results: {results}")
