# multi_agent/QnA_Agent/agent.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain import hub

load_dotenv(dotenv_path="../../.env")

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')


def qna_agent_response(user_input: str) -> str:
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    search_tool = DuckDuckGoSearchRun()
    tool = Tool(
        name="Search",
        func=search_tool.run,
        description="Use this tool to look up information on the web ONLY when needed."
    )

    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm=llm, tools=[tool], prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[tool],
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    result = agent_executor.invoke({"input": user_input})
    return result['output']
