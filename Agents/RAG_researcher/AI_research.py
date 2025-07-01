from dotenv import load_dotenv
import os
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Load environment variables
load_dotenv(dotenv_path="../../.env")

# Step 1: Configure Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", google_api_key=os.environ.get("GOOGLE_API_KEY")
)


# Step 2: Define PDF summarizer tool globally
@tool
def pdf_summarizer(query: str) -> str:
    """
    Loads the PDF file, splits it into chunks, and retrieves relevant information based on the query.
    """
    
    if not path:
        return "No PDF file provided. Please try using web search or Wikipedia."

    try:
        loader = PyPDFLoader(file_path=path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embedding_model)

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        results = retriever.invoke(query)

        return "\n\n".join([doc.page_content for doc in results])

    except Exception as e:
        return f"Error processing PDF: {e}"


# Step 3: Web and Wikipedia tools
#web_search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Step 4: Prompt and Agent Setup
prompt = hub.pull("hwchase17/react")


@tool
def research_agent(query: str, pdf_file_path: Optional[str] = None) -> str:
    """This is a simple AI assistant that can:
    - Search the web using DuckDuckGo
    - Get info from Wikipedia
    - Summarize your uploaded PDF documents

    It uses LangChain and Google Gemini LLM (2.0 Flash) to work like a mini research assistant.
    ## Tools Used"""
    global path

    # Step 6: Prepare tools
    tools = [ wiki]
    if pdf_file_path:
        path = pdf_file_path  # Assign to global variable for tool to access
        tools.append(pdf_summarizer)

    # Step 7: Create and run agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    input_dict = (
        {"input": query, "pdf_path": pdf_file_path}
        if pdf_file_path
        else {"input": query}
    )
    response = agent_executor.invoke(input_dict)

    return response["output"]
