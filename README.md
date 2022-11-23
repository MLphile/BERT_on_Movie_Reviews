# Sentiment Analysis on IMDb Movie Reviews

## Intro
Sentiment analysis is a use case of text classification which consists of assigning a category to a given text. It's a powerful Natural Language Processing (NLP) technique that makes it possible to automatically analyze what people think about a certain topic. This can help companies and individuals to quickly make more informed decisions. Sentiment analysis has for example applications in [social media, customer service and market research](https://www.taus.net/resources/blog/what-is-sentiment-analysis-types-and-use-cases).  
The aim of this project is to build a model that can accurately predict whether a movie review is positive or not. This could for example help to automatically summarize all the reviews for a given movie and therefore help users to make a decision quickly without wasting time reading all the reviews.
This project is about text classification using an [IMDB daset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) consisting of about 50K movie reviews.  
We'll fine-tune BERT(Bidirectional Encoder Representations from Transformers) to predict whether a review is positive or not, and then build a simple streamlit app out of it.

## Data

## Model and Training
For this project I leveraged the pretrained **Hugging Face Bert Model** (bert-base-uncased). The model was fine-tuned using the following hyper-parameters:
* `Learning rate = 2e-5` using `AdamW` optimizer
* `Linear scheduler` with `num_warmup_steps = 0`
* `Maximum sequence length = 128`
* `Batch size = 32`
* `Number of training epochs = 5`

[Here](https://huggingface.co/docs/transformers/training) is a nice tutorial from Hugging Face explaining how to fine-tune a pretrained model.
## Evaluation
The accuracy on the validation set reached 89.35. Evaluation results on the test can be seen in the table below.
| class | precision | recall | f1-score | support |
| --- | --- | --- |--- |--- |
| 0 | 0.91 | 0.89 | 0.90 | 3705 |
| 1 | 0.90 | 0.92 | 0.91 | 3733 |

## Project Structure

## How to run

## References
* [What is Sentiment Analysis? Types and Use Cases](What is Sentiment Analysis? Types and Use Cases)
* [Hugging Face](https://huggingface.co/docs/transformers/index)
