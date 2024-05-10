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
  
