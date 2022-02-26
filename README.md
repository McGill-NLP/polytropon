# polytropon

## Installation

```python
git clone https://github.com/McGill-NLP/polytropon.git
cd polytropon
pip install -e .
```

## Usage

```python
from polytropon import SkilledModel

model = SkilledModel.from_pretrained(
  pretrained_model_name_or_path,
  n_tasks,
  n_skills,
)
```
