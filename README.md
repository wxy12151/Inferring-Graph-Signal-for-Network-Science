```diff
+ QUICKSTART:
- FORK THIS REPOSITORY OR USE IT AS TEMPLATE
- RENAME THE REPOSITORY WITH THE TITLE OF YOUR PUBLICATION
- ADD THE PUBLICATION TO THE UCL IRIS DATABASE
- ALLOW THE PAPER OPENACCESS
- REMOVE THIS CODE BLOCK FROM THE NEWLY FORKED/TEMPLATED REPOSITORY
```

[**Prerequisites**](#prerequisites)
| [**Install guide**](#install-guide)
| [**Experiments**](#experiments)
| [**Contacts**](#contacts) 
| [**Cite**](#cite)

# Computing Machinery and Intelligence
#### Author(s):
Alan Turing, Victoria University of Manchester, Department of Mathematics
#### Manuscript:
https://academic.oup.com/mind/article-pdf/LIX/236/433/30123314/lix-236-433.pdf
#### Abstract 
_I propose to consider the question, “Can machines think?”♣ This should begin with definitions of the meaning of the terms “machine” and “think”. The definitions might be framed so as to reflect so far as possible the normal use of the words, but this attitude is dangerous. If the meaning of the words “machine” and “think” are to be found by examining how they are commonly used it is difficult to escape the conclusion that the meaning and the answer to the question, “Can machines think?” is to be sought in a statistical survey such as a Gallup poll._

## Prerequisites
Explain the prerequisites for the installation here, such as the OS, or whether the program requires a GPU.
This is **not** a place to add python (or language) requirements, please see the [install guide](#install-guide) below.
An example is below:
* Ubuntu Linux > 16.04
* Anaconda python > 3.0.5

## Install guide
Make sure that your environment is reproducible.
You can use a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-file-manually) to specify the python version and the [pip requirements](https://stackoverflow.com/a/35245610/6655465).

```sh
conda install environment.yml
```

## Experiments
It can be useful to pack experiments into python modules.
Add the code required to run a specific experiment - an example is shown below.
```sh
python experiments/the-imitation-game.py
```

## Contacts
Add your contact details here, included but not limited to your email, twitter, google scholar, GitHub.
* alan@turing.ai
* https://twitter.com/aturing
* https://scholar.google.co.uk/citations?user=VWCHlwkAAAAJ&hl=en&oi=sra
* https://github.com/aturing


## Cite
```
@article{turing1950mind,
  title={Mind},
  author={Turing, Alan Mathison},
  journal={Mind},
  volume={59},
  number={236},
  pages={433--460},
  year={1950}
}

```
