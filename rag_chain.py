import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_llm_and_prompt():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer using only the context below.
If the answer is not in the context, say 'I do not have enough information to answer that.'

Context:
{context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    return llm, prompt

if __name__ == "__main__":
    print("Loading RAG chain...")
    retriever = load_retriever()
    llm, prompt = build_llm_and_prompt()
    print("Ready! Type your questions (type 'quit' to exit)\n")

    chat_history = []

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit"]:
            break
        if not question:
            continue

        docs = retriever.invoke(question)
        context = format_docs(docs)

        messages = prompt.format_messages(
            context=context,
            chat_history=chat_history,
            question=question
        )

        response = llm.invoke(messages)
        answer = response.content

        print(f"\nBot: {answer}")

        if docs:
            pages = sorted(set([
                doc.metadata.get("page", 0) + 1
                for doc in docs
                if isinstance(doc.metadata.get("page"), int)
            ]))
            print(f"Sources: pages {pages}")
        print()

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))