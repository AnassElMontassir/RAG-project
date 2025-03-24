# app.py

import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os


#CONFIG
#modèle pour transformer les textes en vecteurs (basés sur similarité sémantique entre les textes <=> similarité cosinus entre les vecteurs)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#dossier où l’index vectoriel est stocké
INDEX_DIR = "faiss_index"

#modèle LLM qu'on utilisera, il est hébergé sur huggingface
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"



#CHARGEMENT DES EMBEDDINGS 
def load_index(index_dir):
    """
    charge l’index FAISS à partir du disque.  
    utilise le modèle d'embedding défini dans la config.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

def get_qa_chain(vectorstore):
    """
    prépare la chaîne de questions/réponses en connectant un LLM
    au système de récupération de documents (retriever).
    """
    llm = HuggingFaceHub(
        repo_id=HF_MODEL,
        model_kwargs={"temperature": 0.3, "max_new_tokens": 1024},
    )
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

#interface avec streamlit
st.set_page_config(page_title="Book Q&A RAG", layout="wide")
st.title("RAG sur Harry Potter & Hunger Games")

with st.spinner("Chargement de l'index vectoriel..."):
    if not os.path.exists(INDEX_DIR):
        st.error(f"Dossier d'index introuvable : {INDEX_DIR}. Veuillez d'abord exécuter le script d'indexation.")
        st.stop()
    db = load_index(INDEX_DIR)
    qa = get_qa_chain(db)

st.success("Index chargé ! Pose ta question.")
question = st.text_input("Ta question sur un ou plusieurs livres :")

if question:
    with st.spinner("🧠 Recherche de la réponse..."):
        result = qa({"query": question})

        st.subheader("📌 Réponse :")
        st.write(result['result'])

        st.subheader("📚 Sources :")
        for doc in result['source_documents']:
            st.markdown(f"**{doc.metadata['source']}** : `{doc.page_content[:200]}...`")
else:
    st.info("Pose une question pour interroger les livres chargés par défaut.")
