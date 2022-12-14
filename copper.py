import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

#use this to paraphrase sentences

text = "A well-organized paragraph supports or develops a single controlling idea, which is expressed in a sentence called the topic sentence. "
get_response(text, 2)



#use this to parapharase entire paragraphs

context = "paste sentence here"
print(context)

from sentence_splitter import SentenceSplitter, split_text_into_sentences

splitter = SentenceSplitter(language='en')

sentence_list = splitter.split(context)
sentence_list

paraphrase = []

for i in sentence_list:
  a = get_response(i,1)
  paraphrase.append(a)

paraphrase

paraphrase2 = [' '.join(x) for x in paraphrase]
paraphrase2

paraphrase3 = [' '.join(x for x in paraphrase2) ]
paraphrased_text = str(paraphrase3).strip('[]').strip("'")
paraphrased_text

print(context)
print(paraphrased_text)

