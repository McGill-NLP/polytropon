# Polytropon: Combining Modular Skills in Multitask Learning

The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages. The dataset is the translation and reannotation of the English [**COPA**](https://people.ict.usc.edu/~gordon/copa.html) ([**Roemmele et al. 2011**](#cite)) and covers 11 languages from 11 families and several areas around the globe. The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages. All the details about the creation of XCOPA and the implementation of the baselines are available in the [**paper**](#).

[**Installation**](#installation) | [**Usage**](#usage) | [**Cite**](#cite) | [**Paper**](https://github.com/McGill-NLP/polytropon/media/paper.pdf)


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
