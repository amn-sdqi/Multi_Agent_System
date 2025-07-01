from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_core.output_parsers import StrOutputParser



from Agents.Automation_agent.automation_agent import (
    email_agent,
    report_agent,
)
from Agents.RAG_researcher.AI_research import research_agent
from Agents.QnA_Agent.qna_agent import qna_agent_response
from Agents.Sentiment_Agent.agent_core import analysis_tool
from Agents.data_analysis.data_analysis_agent import data_analysis
from Agents.stock_agent.main import main

from dotenv import load_dotenv

load_dotenv()

primary_memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)
# prompt = PromptTemplate(
#     input_variables=["input"],
#     template="Choose tools wisely based on the input query: {input}"
# )
prompt = hub.pull("hwchase17/react-chat")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")  

parser = StrOutputParser()

tools = [
    Tool(
        name="EmailAgent",
        func=email_agent(),
        description="draf and Send emails , also email asistince",
    ),
    Tool(name="ReportAgent", func=report_agent(), description="Generate reports"),
    Tool(
        name="analysis_tool",
        func=analysis_tool,
        description="Analyze the input text and return sentiment + primary emotion.",
    ),
    Tool(
        name="QnA_Agent",
        func=qna_agent_response,
        description="Answer the input query and retain the context for future use",
    ),
    Tool(name="AI_investment",
          func=main,      
          description="Analyses the stock market and investment options"),
    Tool(
        name="AI_research_assistant",
        func=research_agent,
        description="""This is a simple AI assistant that can:
                        - Search the web using DuckDuckGo
                        - Get info from Wikipedia
                        - Summarize your uploaded PDF documents

                        It uses LangChain and Google Gemini LLM (2.0 Flash) to work like a mini research assistant.
                        ## Tools Used""",
    ),
    Tool(name="NotifyAgent", func=notify_run, description="Send notifications"),
    Tool(
        name="data_analysis",
        func=data_analysis,
        description="Run data analysis on the input query and return results",
    ),
]


agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=primary_memory,
    verbose=False,
    handle_parsing_errors=True,
)
while True:    
    query = input("Enter your demands: ")
    if query in ['quit', 'exit']:
        break
    result = agent_executor.invoke({"input": query})
    print(result['output'])
