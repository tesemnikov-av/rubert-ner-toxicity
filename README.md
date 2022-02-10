![contributors](https://img.shields.io/github/contributors/tesemnikov-av/rubert-ner-toxicity) ![last-commit](https://img.shields.io/github/last-commit/tesemnikov-av/rubert-ner-toxicity) ![repo-size](https://img.shields.io/github/repo-size/tesemnikov-av/rubert-ner-toxicity)

# RuBert NER Toxicity
<img src="logo.png" width="150"/>

Fine-tuning cointegrated/rubert-tiny-toxicity model on data from toxic_dataset_ner.
How i prepared dataset and fine-tuning the model, see [here.](https://github.com/tesemnikov-av/rubert-ner-toxicity/blob/main/Toxic_Russian_Comments.ipynb).

<img src="img.png" width="700"/>

```python
from ipymarkup import show_span_box_markup

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    pipeline
)

model = AutoModelForTokenClassification.from_pretrained('tesemnikov-av/rubert-ner-toxicity')
tokenizer = AutoTokenizer.from_pretrained('tesemnikov-av/rubert-ner-toxicity')

pipe = pipeline(model=model, tokenizer=tokenizer, task='ner', aggregation_strategy='average')

text = "Вот дурак то! И надо-же быть таким придурком!"
spans = pipe(text.lower())

spans_list = []
for span in spans:
    spans_list.append((span['start'], span['end'], span['entity_group']))
    
show_span_box_markup(text, spans_list)
```
