from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from rag.state import GraphState

class RAGGraph:
    def __init__(self,retriever_chain):
        self.retriever=retriever_chain
        self.llm=ChatGroq(
            model='llama-3.1-70b-versatile',
            temperature=0
            )
        self.app=self.build_graph()

    def retriever_node(self, state:GraphState):
        question=state['question']
        document=self.retriever.invoke(question)
        return ({'document':document})
    def generation_node(self, state:GraphState):
        question=state['question']
        document=state['document']

        context_text='\n\n'.join([doc.page_content for doc in document])

        template="""
        Bạn là trợ lý AI. Hãy trả lời câu hỏi CHỈ DỰA TRÊN THÔNG TIN TRONG LUẬT DƯỚI ĐÂY.
        Nếu không có thông tin trong Context, hãy trả lời:
        "Tôi không tìm thấy thông tin trong tài liệu."

        Context:
        {context}

        Câu hỏi:
        {question} 
        """

        prompt=ChatPromptTemplate.from_template(template)

        chain= prompt | self.llm | StrOutputParser()

        answer=chain.invoke({'question':question,'context':context_text})

        return ({'generation':answer})
    
    def build_graph(self):
        workflow=StateGraph(GraphState)

        workflow.add_node('retriever',self.retriever_node)  
        workflow.add_node('generation',self.generation_node)

        workflow.set_entry_point('retriever')
        workflow.add_edge('retriever','generation')
        workflow.add_edge('generation',END)     
        
        return workflow.compile()