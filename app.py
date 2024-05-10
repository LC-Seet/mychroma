# import
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter

def main():
    # load the document and split it into chunks
    loader = TextLoader("./state_of_the_union.txt")
    documents = loader.load()

    print(documents)

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = text_splitter.split_documents(documents)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    db = Chroma.from_documents(docs, embedding_function)

    # query it
    query = "What did the president say about Ketanji Brown Jackson"
    docs = db.similarity_search(query)

    # print results
    print(docs[0].page_content)

    print(docs[0].metadata)

    # save to disk
    db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
    docs = db2.similarity_search(query)


if __name__ == "__main__":
    main()
  
