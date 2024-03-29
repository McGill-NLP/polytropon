# Polytropon: Combining Modular Skills in Multitask Learning

[**Updates**](#updates) | [**Installation**](#installation) | [**Usage**](#usage) | [**Cite**](#cite) | [**Paper**](media/paper.pdf)

## Updates

23/05/03: *New repository!* The current repository is outdated. We recommend to use instead the full implementation of Polytropon at https://github.com/microsoft/mttl

## Installation

```python
pip install git+https://github.com/McGill-NLP/polytropon
```

Otherwise, if you wish to clone the repo:

```python
git clone https://github.com/McGill-NLP/polytropon.git
cd polytropon
pip install -e .
```

## Usage

```python
from polytropon import SkilledMixin

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

## Cite

```
@misc{ponti2022combining,
      title={Combining Modular Skills in Multitask Learning},
      author={Edoardo M. Ponti and Alessandro Sordoni and Yoshua Bengio and Siva Reddy},
      year={2022},
      eprint={2202.13914},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
