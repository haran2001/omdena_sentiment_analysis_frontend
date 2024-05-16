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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForSequenceClassification,
)


from accelerate import init_empty_weights
from transformers import BertConfig, BertModel
from pysentimiento import create_analyzer


st.set_page_config(layout="wide")

device = "cuda" if torch.cuda.is_available() else "cpu"


def add_logo():
    st.image("assets/images/Omdena.png", width=250)


@st.cache_resource
def load_model():
    MODEL = f"VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis"
    config = BertConfig.from_pretrained(MODEL)
    model = BertForSequenceClassification.from_pretrained(MODEL)
    tokenizer = BertTokenizer.from_pretrained(MODEL)

    # MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    # tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # config = AutoConfig.from_pretrained(MODEL)
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL)

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
    # sa_bert_model, sa_bert_tokenizer = load_model()
    analyzer = create_analyzer(task="sentiment", lang="es")

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
            "<h4 style color:black;'>robertuito-sentiment-analysis</h4>",
            unsafe_allow_html=True,
        )
        user_input1 = st.text_input("", key="input1")
        if st.button("Classify RoBERTa"):
            # predicted_label, probabilities = predict(
            #     sa_bert_model, sa_bert_tokenizer, user_input1
            # )
            # st.write(predicted_label)
            # st.write(probabilities)
            # st.write("Note: True is positive sentiment, False is negative sentiment")

            prediction = analyzer.predict("Qué gran jugador es Messi")
            st.write(prediction.output)
            st.write(prediction.probas)
            st.write("Note: POS is positive, NEG is negative, NEU is neutral")


if __name__ == "__main__":
    main()
