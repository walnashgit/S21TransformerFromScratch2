# GPT2 Trained on Tiny Shakespeare text.
## Model follows Andrej Karpathy's [video](https://www.youtube.com/watch?v=l8pRSuU81PU) on GPT2

## Usage

### Clone the repo in a jupityer note book or where ever you are going to train:
```
!git clone https://github.com/walnashgit/S21TransformerFromScratch2.git
```

### Move to the root folder and install the requirements:

```
%cd /content/S21TransformerFromScratch2
!pip install tiktoken
```

### for training just run the below in notebook
```
from train import *
```

### to save taining model
```
torch.save(model.state_dict(), "S21_model.pth")
```

### For training logs see the [notebook](https://github.com/walnashgit/S21TransformerFromScratch2/blob/main/S21TransformerFromScratch2.ipynb) file in the repo.

### Files


1. input.text: Tiny Shakerpeare training data.
2. model.py: GPT2 model
3. train.py: training code
4. util.py: support util functions
5. S21TransformerFromScratch2.ipynb: Jupyter notebook for training.


### Hugging face demo app: 
[App](https://huggingface.co/spaces/walnash/ERAV2-21-GPT2)

![image](https://github.com/walnashgit/S21TransformerFromScratch2/assets/73463300/ee3f5161-af31-4408-9b9c-028222311676)
