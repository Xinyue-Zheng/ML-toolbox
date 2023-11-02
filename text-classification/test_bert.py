from transformers import BertConfig, BertModel, AutoTokenizer, BertForPreTraining


# Initializing a BERT bert-base-uncased style configuration

configuration = BertConfig()

# Initializing a model (with random weights) from the bert-base-uncased style configuration

model = BertForPreTraining.from_pretrained('bert-base-uncased')

# Accessing the model configuration

# using the AutoTokenizer to initialize the text
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sequence = "In a hole in the ground there lived a hobbit."

inputs = tokenizer(sequence, return_tensors='pt')

configuration = BertConfig()
model1=BertModel.from_pretrained('bert-base-uncased')
model2=BertModel(configuration)

outputs1=model1(**inputs)
outputs2=model2(**inputs)

outputs = model(**inputs)
#
#
# from transformers import BertTokenizer, BertLMHeadModel
# import torch
# from torch.nn import functional as F
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertLMHeadModel.from_pretrained('bert-base-uncased', is_decoder=True)
# text = "A knife is very "
# input = tokenizer(text, return_tensors="pt")
# model_output=model(**input)
# output = model(**input).logits[:, -1, :]
# softmax = F.softmax(output, -1)
# index = torch.argmax(softmax, dim=-1)
# x = tokenizer.decode(index)
# print(x)


from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
text = "The capital of France, " + tokenizer.mask_token + ",contains the Eiffel Tower."
input = tokenizer.encode_plus(text, return_tensors="pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)
output = model(**input)
logits = output.logits
softmax = F.softmax(logits, dim=-1)
mask_word = softmax[0, mask_index, :]
top_word = torch.argmax(mask_word, dim=1)
print(tokenizer.decode(top_word))

# from transformers import BertTokenizer, BertLMHeadModel, BertConfig
# import torch
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# config = BertConfig.from_pretrained("bert-base-uncased")
# config.is_decoder = True
# model = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)
# inputs = tokenizer("Hello, my dog [MASK] cute", return_tensors="pt")
# inputs2 = tokenizer("Hello, my dog is cute", return_tensors="pt")
# label_id = inputs['input_ids'] == inputs2['input_ids']
# label = inputs2['input_ids'].clone()
# label[label_id] = -100
# outputs = model(**inputs,labels=label)
# prediction_logits = outputs.logits
# loss = outputs.loss


from transformers import BertTokenizer, BertForNextSentencePrediction
from torch.nn import functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
prompt = "The child came home from school."
next_sentence = "He played soccer after school."
encoding = tokenizer(prompt, next_sentence, return_tensors='pt')
output_model=model(**encoding)
outputs = model(**encoding)[0]
softmax = F.softmax(outputs, dim=1)
print(softmax)

import torch

from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():

    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()

model.config.id2label[predicted_class_id]