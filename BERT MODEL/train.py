from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from preprocess import preprocess_data

train_data, test_data = preprocess_data()
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,   
    warmup_steps=500,                
    logging_dir='./logs',            
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the model to train
    args=training_args,                  # training arguments
    train_dataset=train_data,            # training dataset
    eval_dataset=test_data,              # evaluation dataset
    compute_metrics=compute_metrics      # evaluation metrics
)

trainer.train()
