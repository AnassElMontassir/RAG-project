# 📚 RAG - Retrieval-Augmented Generation on Books

This project is a Streamlit-based interactive application that allows you to ask questions about three popular books (Harry Potter 1 & 2, Hunger Games), using a Retrieval-Augmented Generation (RAG) system with an open-source LLM.

---

## 🚀 Features

- 🔍 Semantic search within book content
- 🤖 Response generation using an LLM (Mistral-7B-Instruct)
- 📚 Source references for each answer
- 💡 Cross-book questions supported

---

## 📁 Project Structure

```
.
├── app.py                # Streamlit application
├── build_vectorstore.py        # Script to build the FAISS index
├── data/                # Folder containing book files (.pdf or .md)
├── faiss_index/          # Saved vector index (created after running build_vectorstore.py)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/book-rag-project.git](https://github.com/AnassElMontassir/RAG-project.git
cd RAG-project
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv RAG_Env
source RAG_Env/bin/activate  # or RAG_Env\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🔑 Hugging Face API Configuration

The LLM is hosted on [Hugging Face Hub](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).  
Create a token at: https://huggingface.co/settings/tokens

Then set the environment variable:

```bash
export HUGGINGFACEHUB_API_TOKEN=your_token_here  # Linux/macOS
set HUGGINGFACEHUB_API_TOKEN=your_token_here     # Windows CMD
```

---

## 🛠️ Build the FAISS index

Run this first before running the application:

```bash
python build_vectorstore.py
```

This will:
- Read all files in `data/`
- Split them into chunks
- Generate embeddings for every chunk
- Save the FAISS index to `faiss_index/`

---

## ▶️ Run the application

```bash
streamlit run app.py
```

## 🧩 How it works

1. **Index Creation** (via `build_vectorstore.py`):
   - Each book is split into semantic **chunks** (2000 character each).
   - Chunks are encoded into vectors using a **sentence-transformers** model.
   - All vectors are stored in a **FAISS** index along with their source metadata.

2. **Question Answering** (via `app.py`):
   - The user enters a **natural language question** in the Streamlit interface.
   - The question is encoded into a vector using the **same embedding model**.
   - The system retrieves the **most relevant chunks** from the FAISS index.
   - These chunks are passed to an **LLM (Mistral-7B-Instruct)** with the user question.
   - The LLM generates a **context-aware answer** based only on the retrieved text.
   - The app also displays the **source documents** used in the answer.

This architecture ensures factual answers grounded in the content of the uploaded books, while keeping generation controlled and interpretable.

---

## 🧪 Example Questions

**Single-book questions:**
- What time did the "Poudlard Express" leave? *(Harry Potter 1)*
- Why did Hermione lock herself in the bathroom? *(Harry Potter 1)*

**Cross-book questions:**
- Who is older between Harry and Katniss? *(Harry Potter & Hunger Games)*
- What do Harry Potter's and Katniss' fathers have in common? *(Harry Potter & Hunger Games)*


---

## 🧠 Tech Stack

- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Hugging Face Hub](https://huggingface.co/)
- [Streamlit](https://streamlit.io/)
- Embedding Model: [`sentence-transformers/all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- LLM: `mistralai/Mistral-7B-Instruct-v0.1`

---

## 📄 License

This project is open-source and educational — feel free to fork, modify, and reuse it as a base for your own RAG apps.

---
