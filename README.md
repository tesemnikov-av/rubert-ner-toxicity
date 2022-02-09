# RuBert NER Toxicity

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
![toxic]('./img.png')
<img src="img.png" width="150"/>
