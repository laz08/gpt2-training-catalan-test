
<!-- omit in toc -->
# GPT-2 in Catalan

This repository serves as a place to document a toy attempt on how to create a generative text model in Catalan, based on GPT-2.
In other words... this is more of a prototype and a personal playground than a serious attempt to have a fully functional GPT-2 in Catalan. 

Nevertheless, I hope this can also help someone else train their own GPT-2 model and provide some pointers on how to do so.

Suggestions and constructive criticism are always welcome!

- [1. GPT-2 :memo:](#1-gpt-2-memo)
  - [1.1. What is GPT-2 :question:](#11-what-is-gpt-2-question)
  - [1.2. Why GPT-2 :grey_question:](#12-why-gpt-2-grey_question)
- [2. Training :hammer:](#2-training-hammer)
  - [2.1. Requirements :paperclip:](#21-requirements-paperclip)
  - [2.2. Training Script :chart_with_upwards_trend:](#22-training-script-chart_with_upwards_trend)
    - [2.2.1. Training GPT-2 in Catalan](#221-training-gpt-2-in-catalan)
  - [2.3. About the data used for Catalan :open_file_folder:](#23-about-the-data-used-for-catalan-open_file_folder)
    - [2.3.1. WikiCorpus PROs :thumbsup:](#231-wikicorpus-pros-thumbsup)
    - [2.3.2. WikiCorpus CONs :thumbsdown:](#232-wikicorpus-cons-thumbsdown)
  - [2.4. Further training for specific tasks :zap:](#24-further-training-for-specific-tasks-zap)
- [3. Testing the model :cat:](#3-testing-the-model-cat)
- [4. Questions  :question: :grey_question:](#4-questions--question-grey_question)
  - [4.1. Why Catalan :question:](#41-why-catalan-question)
  - [4.2. Why use a Pretrained model in Spanish :grey_question:](#42-why-use-a-pretrained-model-in-spanish-grey_question)
  - [4.3. Can I use another data/language :question:](#43-can-i-use-another-datalanguage-question)
- [5. TO-DO :construction:](#5-to-do-construction)



# 1. GPT-2 :memo:
## 1.1. What is GPT-2 :question:
GPT-2 (GPT-2 stands for _Generative Pre-trained Transformer 2_) is a transformer-based language model trained in large volumes of data and was not trained with a specific task in mind. Nevertheless, it has probably been used mostly for generating new text.

A better and further explanation can be found [here (http://jalammar.github.io/illustrated-gpt2/)]([here](http://jalammar.github.io/illustrated-gpt2/)).

## 1.2. Why GPT-2 :grey_question:

It is undeniable that GPT-2 played a large role and became very popular when it came out. It has also created some controversy.
These aside, GPT-2 acted as a big step forward in terms of generating texts... And is also "faster" to train on custom data than its next generation sibling, GPT-3.

# 2. Training :hammer:

## 2.1. Requirements :paperclip:
You will need a powerful GPU or reduce the batch size. You can also use a VM from a Cloud service such as Google Colab or Microsoft Azure.

## 2.2. Training Script :chart_with_upwards_trend:
The training is implemented in the `train_GPT2.py` script, which serves as a skeleton. You can run it from the Commandline and passing all the arguments.


e.g. 
```bash
cd src
./train_GPT2.py \
    --model <baseModel> \
    --tokenizer <baseTokenizer> \
    --train_path <path_to_training_data> \
    --test_path <path_to_test_data> \
    --n_epochs <number_of_epochs> \
    --train_batch_size <size_batch_for_training> \
    --eval_batch_size <size_batch_for_evaluation> \
    --eval_steps <n_steps_during_training_for_evaluation> \
    --save_steps <n_steps_checkpoint_save> \
    --warmup_steps <warm_up_steps> \
    --output <model_output_name>
```
### 2.2.1. Training GPT-2 in Catalan

Taking advantage of the fact that Spanish is pretty close to the Catalan language, we may take its tokenizer.

Nevertheless, as a TODO, we should create a tokenizer especificly for Catalan.

We can also take advantage of previously trained models in Spanish and further train on them with data written in Catalan.

For example:
```bash
cd src
./train_GPT2.py \
    --model DeepESP/gpt2-spanish \
    --tokenizer DeepESP/gpt2-spanish \
    --train_path ../data/catalan_corpus_train.csv \
    --test_path ../data/catalan_corpus_test.csv \
    --n_epochs 1 \
    --train_batch_size 4 \
    --eval_batch_size 8 \
    --eval_steps 100 \
    --save_steps 1000 \
    --warmup_steps 300 \
    --output gpt2-catalan
```
Note: Of course, the higher the batch size during training, the better. Change this accordingly to the GPU you have.
Training on a CPU is also possible, but it is not advised at all, for it will take way longer.


## 2.3. About the data used for Catalan :open_file_folder:
The data used has mostly been the [WikiCorpus](https://www.cs.upc.edu/~nlp/wikicorpus/) data provided by the [Computer Science](https://www.cs.upc.edu/) department @ FIB, UPC (Facultat d'Informàtica de Barcelona, Universitat Politècnica de Catalunya).

You can download it using the `datasets` library from Huggingface:
```python
from datasets import load_dataset

dataset = load_dataset("wikicorpus, 'raw_ca')
```

Or you can use the `download_wikicorpus.py` file in this repository, which also splits  the data in train/test and can create a smaller subset for testing, if desired.

### 2.3.1. WikiCorpus PROs :thumbsup:
Well, the data is already obtained. That's always a pro.

### 2.3.2. WikiCorpus CONs :thumbsdown:
We are limiting the knowledge of the Language model to data from the Wikipedia. Therefore, this model will probably be more error-prone with informal text inputs. This includes data from chats, colloquialisms and text from social media.

Additionally, the size of the data is tiny with respect to what it should be.

## 2.4. Further training for specific tasks :zap:
Once the model is trained in Catalan and we have a base, we can further train this model for a specific task in mind.

A couple of Proof of Concepts (PoC) have been done using data gathered from Twitter and also from Catalan songs.

# 3. Testing the model :cat:
We can test the trained model easily using the script `test_generation.py`.
```bash
cd src
python .\test_generation.py -t DeepESP/gpt2-spanish -m ../data/gpt2-catalan -i generation_test.txt
```

# 4. Questions  :question: :grey_question:
## 4.1. Why Catalan :question:
Artificial Intelligence should not be only for largely spoken languages, such as English or even Spanish.
Catalan, a minority language, is my mother tongue and it's always fun to see something you work with also operating in your own language. So why not?


## 4.2. Why use a Pretrained model in Spanish :grey_question:
Although Spanish and Catalan are different languages, they share a lot of expressions, vocabulary and grammatical structures. Therefore, basing a Catalan model on a previously trained model in a close language such as Spanish is not unreasonable. 

Transferring the knowledge from it to our model is better than starting from zero, specially to save computational time.


## 4.3. Can I use another data/language :question:
Even though the scripts are all prepared with the project of training the model with data in Catalan in mind, the scripts should work with any text data, be it Catalan from the Wikicorpus or texts from songs, books, etc.


Feel free to change the `CatalanDataset` class or swap it with yours, since probably formatting of the input text is the most varying aspect between projects.

Be sure to also change the base model, since if you want to train another language (e.g. German), basing it on a pre-trained model in Spanish will not work well.



# 5. TO-DO :construction:

- Create our own Tokenizer.
  
- Train from scratch. Since we are actually using the Transfer learning approach and relying on a previously pretrained model in Spanish, we probably don't have as an accurate model as we should.

- Increase the volume of data. More varied data should also be used during the training, because it is very biased towards informative data (for obvious reasons).