# pegasus.paraphrase
A paraphrasing tool which uses Artificial Intelligence & NLP (Natural Language Processing) to understand and curate new sentences(or paragraphs)


Start by using a notebook (jupyter or https://colab.research.google.com/). I have used https://colab.research.google.com/ here. 

Installing Modules 

```bash
pip install sentence-splitter
```


```bash
pip install transformers
```


```bash
pip install sentencepiece
```


Make sure you have torch installed (pytorch). 

If not run 

```bash
pip install torch
```


Go to https://colab.research.google.com/ make a new file. Go to the runtime option from the nav bar, select change runtime type and change hardware accelator as GPU.

Start by entering the following lines of code and running it one by one using the play icon adjacent to the code.

```bash
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
```

```bash
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
```

The above command will download some files and might take some time. 

```bash
def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
```

```bash
def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text
```

Enter the sentence in the following command

```bash
text = "Enter sentence here "
```


In the following command, the number of paraphrased sentenses is set to 2. You can change it as per your requirement. 

```bash
get_response(text, 2)
```


# Just in case you want to paraphrase paragraphs and not sentences alone. Do not run the last two lines of code.
starting from "text = "Enter sentence here ""

and enter the following code in place of it 

```bash
context = "Enter paragraph here"
print(context)
```

```bash
from sentence_splitter import SentenceSplitter, split_text_into_sentences

splitter = SentenceSplitter(language='en')

sentence_list = splitter.split(context)
sentence_list
```

```bash
paraphrase = []

for i in sentence_list:
  a = get_response(i,1)
  paraphrase.append(a)
```


```bash
paraphrase
```

```bash
paraphrase2 = [' '.join(x) for x in paraphrase]
paraphrase2
```

```bash
paraphrase3 = [' '.join(x for x in paraphrase2) ]
paraphrased_text = str(paraphrase3).strip('[]').strip("'")
paraphrased_text
```

```bash
print(context)
print(paraphrased_text)
```

The second line is your paraphrased text. 
