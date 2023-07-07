# Caikit NLP

`caikit_nlp` is a [Caikit](https://github.com/caikit/caikit) library that currently provides [PEFT prompt tuning](https://github.com/huggingface/peft) and MPT (multi-task prompt tuning) functionalities.

More information on MPT can be found at: https://arxiv.org/abs/2303.02861

Currently causal language models and sequence-to-sequence models are supported.

#### Notes

- The data model for text generative capabilities is baked into this repository itself at `caikit_nlp/data_model/generation.py`.

## Development guidelines

To contribute code to this repository, kindly fork the repository and create [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) with your contribution. For the contribution to be reviewed, we require unit tests for every new functionality added. We have some checks in place for code formatting and linting. We recommend that contributors ensure checks pass locally before opening pull requests for review. The following commands can be used for assistance:

### Useful commands

1. Build the environment  
`tox` -> will build a virtual environment installing all dependencies.  
To execute commands, you can activate the virtual environments created, example: `source .tox/py39/bin/activate`.

2. Run unit tests  
`tox -e py39 -- tests` -> will execute full test suite with a Python 3.9 test environment. Please replace environment with py38 or py310 as needed.  
`tox -e py39 -- tests/data_model/test_text.py` -> will execute the single unit test specified.

3. Run formatter and format files  
`tox -e fmt` -> will run a formatter and format any code files as needed. The files that were formatted locally will need to be committed. Tests will pass when all files are formatted. 

4. Run linter  
`tox -e lint` -> will run pylint on the files and return a lint score
