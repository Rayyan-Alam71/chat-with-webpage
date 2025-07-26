from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
import streamlit as st
load_dotenv()

# creating the prompt


# setting up an llm
model = ChatOpenAI(model='gpt-4o-mini')


def extract_data(url):

    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content
    except Exception as e:
        return "Error occurred"


# summarising the content

def get_summary(content):
    # create a prompt
    prompt = PromptTemplate(
        template='''
            You are a helpful assistant. You are expert at analysing the content and summarizing it.
            Summarize the given website content.
            Include bullet-points explanation wherever required.

            Content : {content}
        ''',
        input_variables=['content']
    )
    # loading a parser
    parser = StrOutputParser()

    try:
        # creating a chain
        summarize_chain = prompt | model | parser
        result = summarize_chain.invoke({
            'content' : content
        })
        return result
    except Exception as e:
        return "Error occurred"

def rag_chat(summary):
    splitter = RecursiveCharacterTextSplitter()
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    chunks = splitter.create_documents([summary])
    # creating a vector store
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # creating a retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k" : 2})

    return retriever


def invoke_chain(retriever, summary, query):

    rag_prompt = PromptTemplate(
        template='''
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If the context is insufficient , just say you don't know

        Context :  {context}
        Question : {question}
    ''',
    input_variables=['context', 'question']
    )

    try:
        parallel_chain = RunnableParallel({
            'context' : retriever ,
            'question' : RunnablePassthrough()
        })

        main_chain = parallel_chain | rag_prompt | model | StrOutputParser()

        result = main_chain.invoke(query)
        
        return result
    except Exception as e:
        return "error"

# Streamlit UI
st.title("Website Summarization")
st.write("Enter a URL to extract and summarize the website content")

url = st.text_input("Enter website URL")
if "summary" not in st.session_state:
    st.session_state.summary = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "showAsk" not in st.session_state:
    st.session_state.showAsk = False

def main():
    if st.button("Summarize"):
        if url : 
            try:
                with st.spinner("Extracting content..."):
                    text = extract_data(url = url)
                    if text == "Error occurred":
                        st.warning("Incorrect website url")
                        return
                    else:
                        st.success("Content extracted!")
                        showAsk = True
            
                with st.spinner("Summarizing..."):
                    summary = get_summary(content = text)
                    if summary == "Error occurred":
                        st.warning("Error while summarization")
                        return
                    else:
                        retriever = rag_chat(summary=summary)
                        st.success("Content summarized")
                        st.write("Summary ready !")
                        st.write(summary)

                        st.session_state.showAsk = True
                        st.session_state.summary = summary
                        st.session_state.retriever = retriever
            except Exception as e:
                st.error(f"Error occurred : {str(e)}")
        else:
            st.warning("Enter a valid url")
    

    if st.session_state.showAsk:
        query = st.text_input("Chat with the summary")

    if st.button("Ask "):

        if query and st.session_state.retriever:
            
            with st.spinner("Performing vector search"):
                retriever = st.session_state.retriever
                summary = st.session_state.summary
                result = invoke_chain(retriever=retriever, summary=summary,query=query)
                if result == "error":
                    return
                else:
                    st.success("Search completed")
                    st.write(result)
main()