# Sentiment Analysis on IMDb Movie Reviews

## Intro
Sentiment analysis is a use case of text classification which consists of assigning a category to a given text. It's a powerful Natural Language Processing (NLP) technique that makes it possible to automatically analyze what people think about a certain topic. This can help companies and individuals to quickly make more informed decisions. Sentiment analysis has for example applications in [social media, customer service and market research](https://www.taus.net/resources/blog/what-is-sentiment-analysis-types-and-use-cases).   
Have you ever wanted to summarize all the reviews for a given movie to decide if that movie is worth watching without wasting time reading all the reviews?
The aim of this project is to build a model that can accurately predict whether a movie review is positive or not and then make it available through a simple streamlit app.

## Data
The data for training is the [IMDB daset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) which consists of about 50K movie reviews.  
We'll fine-tune BERT() to predict whether a review is positive or not, and then .

## Model and Training
I leveraged the pretrained BERT (Bidirectional Encoder Representations from Transformers) made available by Hugging Face. The model (bert-base-uncased) was fine-tuned using the following hyper-parameters:
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
* [What is Sentiment Analysis? Types and Use Cases](https://www.taus.net/resources/blog/what-is-sentiment-analysis-types-and-use-cases)
* [Hugging Face](https://huggingface.co/docs/transformers/index)
