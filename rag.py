from dotenv import load_dotenv

from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()   
# add a voyage key to your .env file, e.g. VOYAGE_API_KEY=your_api_key
embeddings = VoyageAIEmbeddings(model='voyage-3-large')

texts = [
    'Apple makes very good computers.',
    'I believe Apple is innovative.',
    'I love apples.',
    'I am a fan of MacBooks.',
    'I enjoy oranges.',
    'I like Lenove ThinkPads.',
    'I think pears taste very good.'
]

vector_store = FAISS.from_texts(texts, embedding = embeddings)

# print(vector_store.similarity_search('Apples are my favorite fruit', k=7))
print(vector_store.similarity_search('I love lenovo.', k=3))

