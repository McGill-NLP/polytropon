# polytropon

## Installation

```python
git clone https://github.com/McGill-NLP/polytropon.git
cd polytropon
pip install -e .
```

## Usage

```python
# load any pretrained model from transformers
from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# merge it with polytropon
model = SkilledMixin(
  model,
  n_tasks,
  n_skills,
)
```
