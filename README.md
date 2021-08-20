# Generation of Regular Expressions from Natural Language

This is an attempt to replicate the paper [Neural Generation of Regular Expressions from Natural Language with Minimal Domain Knowledge](https://arxiv.org/pdf/1608.03000.pdf). The paper explores the task of translating natural language queries into regular expressions.

Some examples of the queries they used:
```
lines with words with a capital letter and a vowel
lines ending with a character before the string 'dog'
lines containing a letter or a character
lines with a number before a letter and a lower-case letter
lines containing the string 'dog', 2 or more times
```

These queries are quite contrived, and did not look like the questions in Stackoverflow. Nevertheless, I followed the paper and trained a sequence-to-sequence model with attention. The [best run](https://wandb.ai/kingyiusuen/regex-from-natural-language-training/runs/hsyqhewj/overview)  had a character error rate of 0.27, which is not very satisfying. The paper uses a different evaluation metric called DFA-Equal, which takes into account that there are many ways to write equivalent regular expressions, but the paper did not explain how the metric can be implementated.

## How to Use

Set up a virtual environment and install dependencies:

```
make venv
make install
```

Example command to start training:

```
python training/run_experiment --max_epochs=10 --lr=0.0005
```

This will also download the data if it is your first time running it.

# Acknowledgements

- [keon/seq2seq](https://github.com/keon/seq2seq) for the implementation of seq2seq model
- [nicholaslocascio/deep-regex](https://github.com/nicholaslocascio/deep-regex) for the dataset (this is the official repository of the paper)