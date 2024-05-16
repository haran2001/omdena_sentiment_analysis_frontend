import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import textwrap
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import requests, json, sys


import shutil
from transformers import BertForSequenceClassification, BertTokenizer

from huggingface_hub import login, logout
from transformers import AutoTokenizer, AutoModelForCausalLM


from accelerate import init_empty_weights
from transformers import BertConfig, BertModel

st.set_page_config(layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"

hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Your app code goes here

st.markdown(hide_github_icon, unsafe_allow_html=True)


def add_logo():
    st.image("assets/images/Omdena.png", width=250)


@st.cache_resource
def load_model():

    config = BertConfig.from_pretrained(
        "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    )
    with init_empty_weights():
        model = BertModel(config)

    model = BertForSequenceClassification.from_pretrained(
        "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    )
    tokenizer = BertTokenizer.from_pretrained(
        "VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    )

    return model, tokenizer


def predict(model, tokenizer, text, threshold=0.5):
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

    predicted_class = torch.argmax(logits, dim=1).item()
    if probabilities[predicted_class] <= threshold and predicted_class == 1:
        predicted_class = 0

    return bool(predicted_class), probabilities


@st.cache_resource
def llm_pipeline():
    checkpoint = "MBZUAI/LaMini-T5-223M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        device_map=device,
        torch_dtype=torch.float32,
    )

    pipe = pipeline(
        "text2text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm


def main():
    state = False
    add_logo()
    llm = llm_pipeline()
    sa_bert_model, sa_bert_tokenizer = load_model()

    st.markdown(
        "<h1 style='text-align: center; color:white;'>IREX-Sentiment-Analyzer</h1>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            "<h4 style color:black;'>LLM based sentiment analysis</h4>",
            unsafe_allow_html=True,
        )

        user_input = st.text_input("", key="input")

        # Search the database for a response based on user input and update session state
        if st.button("Classify LLM"):
            system_prompt = (
                "Classify the text into neutral, negative, or positive"
                + ": "
                + user_input
            )
            response = llm(system_prompt, max_length=512, do_sample=True)
            st.write(response)

    with col2:
        st.markdown(
            "<h4 style color:black;'>SaBERT based sentiment analysis</h4>",
            unsafe_allow_html=True,
        )
        user_input1 = st.text_input("", key="input1")
        if st.button("Classify SaBERT"):
            predicted_label, probabilities = predict(
                sa_bert_model, sa_bert_tokenizer, user_input1
            )
            st.write(predicted_label)
            st.write(probabilities)
            st.write("Note: True is positive sentiment, False is negative sentiment")


if __name__ == "__main__":
    main()
