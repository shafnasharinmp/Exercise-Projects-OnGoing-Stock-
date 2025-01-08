from transformers import BertForSequenceClassification, BertTokenizer


model = BertForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = BertTokenizer.from_pretrained('./sentiment_model')
