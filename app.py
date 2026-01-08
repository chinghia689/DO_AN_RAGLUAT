import os
from dotenv import load_dotenv
from rag.load_split import load_split
from rag.embedding import get_embeddings
from rag.retriever import retriever
from rag.chain import built_chain

load_dotenv()
api_key=os.getenv('GOOGLE_API_KEY')
if not api_key:
    raise('Not key')

dir_file=os.path.dirname(os.path.abspath(__file__))
path_folder=os.path.join(dir_file,'data')

doc=load_split(path_folder)
embeddings=get_embeddings()
retrievers=retriever(doc,embeddings)
answer=built_chain(retrievers)

print("Nhập câu hỏi (gõ exit để thoát):")
while True:
    q = input(">> ")
    if q.lower() == "exit":
        break
    print("AI:", answer.invoke(q))
