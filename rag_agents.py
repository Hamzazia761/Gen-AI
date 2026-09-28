from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

load_dotenv()   
# add a voyage key to your .env file, e.g. VOYAGE_API_KEY=your_api_key
embeddings = VoyageAIEmbeddings(model='claude-sonnet-5')

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears taste very good.',
    'I hate bananas.',
    'I dislike rasberries.',
    'I despite mangoes.',
    'I love linux.',
    'I have Windows.'
]

vector_store = FAISS.from_texts(texts, embedding = embeddings)

# print(vector_store.similarity_search('What fruits does the person like?', k=3))
# print(vector_store.similarity_search('What fruits does the person hate?', k=3))

# top 3 similar
retriever = vector_store.as_retriever(search_kwargs={'k':3})

retriever_tool = create_retriever_tool(retriever, name='kb_search', description='search fruit knowledge base for information.')
# sonnet 5 dosen't work with the retriever tool, so we use voyage-3-large instead ot gpt-4.1-mini
agent = create_agent(
    model='claude-sonnet-5',
    tools = [retriever_tool],
    system_prompt = (
        'you are a helpful assistant. for question about macbooks, apples or laptops, first call the kb_search tool to retrieve context, then answer succinctly. maybe you have to use it multiple times before answering'
        )
    )

result = agent.invoke({
    'messages': [{'role': 'user', 'content': 'what three fruits does the peron like and what three fruits does the person dislike?'}]
})

print(result['messages'][-1].content)