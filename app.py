import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain.embeddings import SentenceTransformerEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline


checkpoint= "LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype= torch.float32
)

@st.cache_resource
def llm_pipline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p = 0.95
    )
    local_llm= HuggingFacePipeline(pipeline=pipe)
    return local_llm

from langchain.retrievers import SVMRetriever

# svm_retriever = SVMRetriever.from_documents(all_splits,OpenAIEmbeddings())
# docs_svm=svm_retriever.get_relevant_documents(question)
# len(docs_svm)


# def qa_llm():
#     llm = llm_pipline()
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     ####new_index = db.as_retiriever()
#     qa = 
#     )

def process_answer(instruction):
    response = " "
    instruction = instruction
    qa =qa_llm()
    generated_text =qa(instruction)
    answer = generated_text['result']
    return answer,generated_text
