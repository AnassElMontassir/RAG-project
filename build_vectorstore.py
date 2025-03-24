from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import os
import glob

#CONFIG

#chemin vers les livres
BOOKS_DIR = "data"

#chemin vers l'index
INDEX_DIR = "faiss_index"

#le modèle d'embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#la taille des chunks
CHUNK_SIZE = 2000

#la taille des chevauchements
CHUNK_OVERLAP = 200


#CHARGEMENT DES DOCUMENTS
def load_documents(directory):
    documents = []
    files = glob.glob(os.path.join(directory, "*"))
    for filepath in files:
        filename = os.path.basename(filepath)
        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filepath.endswith(".md"):
            loader = UnstructuredMarkdownLoader(filepath)

        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = filename
        documents.extend(docs)
    return documents

#SPLIT EN CHUNKS
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(documents)

#MAIN
def main():
    print("📖 Chargement des documents depuis", BOOKS_DIR)
    documents = load_documents(BOOKS_DIR)
    print(f"➡️ {len(documents)} documents chargés")

    print("🔪 Découpage en chunks...")
    split_docs = split_documents(documents)
    print(f"➡️ {len(split_docs)} chunks générés")

    print("🧠 Génération des embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(split_docs, embeddings)

    print("💾 Sauvegarde de l'index FAISS dans", INDEX_DIR)
    db.save_local(INDEX_DIR)
    print("✅ Indexation terminée !")

#convention python pour éviter que le code ne s'exécute si le fichier est importé ailleurs mais seulement si on exécute ce fichier
if __name__ == "__main__":
    main()