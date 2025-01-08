from transformers import BertTokenizer
from datasets import load_dataset

def preprocess_data():
    dataset = load_dataset('imdb')

    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    train_data = dataset['train'].map(preprocess_function, batched=True)
    test_data = dataset['test'].map(preprocess_function, batched=True)

    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    return train_data, test_data
