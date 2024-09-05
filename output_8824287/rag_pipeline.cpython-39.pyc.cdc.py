
from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
load_dotenv('./rag_pipeline_env.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI('gpt-4o-mini', **('model',))
embedding = OpenAIEmbedding('text-embedding-3-large', **('model',))
PERSIST_DIR = './vectorEmbeddings'

def initialize_index():
    chunk_size = 128
    Settings.llm = llm
    Settings.embed_model = embedding
    Settings.chunk_size = chunk_size
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader('./rag_data').load_data()
        vector_index = VectorStoreIndex.from_documents(documents, **('documents',))
        vector_index.storage_context.persist(PERSIST_DIR, **('persist_dir',))
    else:
        storage_context = StorageContext.from_defaults(PERSIST_DIR, **('persist_dir',))
        vector_index = load_index_from_storage(storage_context, **('storage_context',))
        return vector_index


def get_context(prompt):
    vector_index = initialize_index()
    retriever = VectorIndexRetriever(vector_index, 5, **('index', 'similarity_top_k'))
    retrieved_nodes = retriever.retrieve(prompt)
    context = '\n'.join((lambda .0: for node in .0:
node.textNone)(retrieved_nodes))
    return context

