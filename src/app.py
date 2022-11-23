import streamlit as st
from config import FINETUNED_CHECKPOINT, BERT_CHECKPOINT, MAX_LEN, MAPPING
from utils import clean_text, preprocess, classify
from transformers import AutoModelForSequenceClassification, BertTokenizer



@st.cache(allow_output_mutation=True)
def load_model_tokenizer(model_checkpoint, tokenizer_checkpoint):
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_checkpoint)

    return model, tokenizer

model, tokenizer = load_model_tokenizer(FINETUNED_CHECKPOINT, BERT_CHECKPOINT)

st.title("Movie Reviews Classifier")
review = st.text_area("Write a movie review")

if st.button('Classify'):

    if review.isspace() or len(review) == 0:
        st.markdown('**No review provided!** Please enter some text and then hit classify')
        
    else:
        preprocessed_review = preprocess(review, tokenizer=tokenizer, max_len=MAX_LEN, clean_text=clean_text)
        out = classify(inputs=preprocessed_review, model=model, mapping= MAPPING)
        st.markdown(f"The review is **{out['Label'] }** with a confidence of **{out['Confidence']*100} %**")
        
