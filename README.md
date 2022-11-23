# Sentiment Analysis on IMDb Movie Reviews

## Intro
Sentiment analysis or text classification is about categorizing textual information into 2 or more classes.
This project is about text classification using an [IMDB daset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) consisting of about 50K movie reviews.  
We'll fine-tune BERT(Bidirectional Encoder Representations from Transformers) to predict whether a review is positive or not, and then build a simple streamlit app out of it.

## Data

## Model
For this project I leveraged the pretrained **Hugging Face Bert Model** (bert-base-uncased). The model was fine-tuned using the following hyper-parameters:
* `Learning rate = 2e-5` using `AdamW` optimizer
* `Linear scheduler` with `num_warmup_steps = 0`
* `Maximum sequence length = 128`
* `Batch size = 32`
* `Number of training epochs = 5`

The accuracy on the validation set reached 89.35. Evaluation results on the test can be seen in the table below:


## Project Structure

## How to run

## References
