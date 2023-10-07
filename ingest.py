from langchain.document_loaders import WebBaseLoader, PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings,HuggingFaceEmbeddings
from langchain.vectorstores import Chroma,FAISS
import os
# from constant import CLIENTS
from sentence_transformers import SentenceTransformer



# persist_directory ="db"

# import chromadb
# clients = chromadb.PersistentClient(path="db")


def main():

    ##Loading document and splitting it
    for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith(".pdf"):
                print(file)

                ##PDF LOADER###
                loader = PDFMinerLoader(os.path.join(root, file))


                #####WEB BASE LOADER#####
                # loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")


    ## WEB 
    # Create the split
    # data = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    # texts = text_splitter.split_documents(data)


    ## PDF
    documents = loader.load()
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)


    def embed_index(doc_list, embed_fn, index_store):
        """Function takes in existing vector_store, 
        new doc_list and embedding function that is 
        initialized on appropriate model. Local or online. 
        New embedding is merged with the existing index. If no 
        index given a new one is created"""
        #check whether the doc_list is documents, or text
        try:
            faiss_db = FAISS.from_documents(doc_list, 
                                    embed_fn)  
        except Exception as e:
            faiss_db = FAISS.from_texts(doc_list, 
                                    embed_fn)
        
        if os.path.exists(index_store):
            local_db = FAISS.load_local(index_store,embed_fn)
            #merging the new embedding with the existing index store
            local_db.merge_from(faiss_db)
            print("Merge completed")
            local_db.save_local(index_store)
            print("Updated index saved")
        else:
            faiss_db.save_local(folder_path=index_store)
            print("New store created...")



    # Create Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Vector_Store
    embed_index(doc_list= texts,
            embed_fn=embeddings,
            index_store='new_index')

if __name__ == "__main__":
    main()