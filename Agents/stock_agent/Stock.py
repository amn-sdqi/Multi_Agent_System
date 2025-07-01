from load_env import load_environment
load_environment()
from Agents.stock_agent.chain.investment_report_chain import generate_full_report

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from Agents.stock_agent.tools.ticker_lookup import TickerLookupTool
from Agents.stock_agent.tools.query_parser import extract_valid_companies


import ast
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv(dotenv_path="../../.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# ----------------------------- Helper Functions ----------------------------- #


def extract_valid_companies(user_query: str) -> str:
    """
    Given a user query, use the LLM to extract a list of valid company names.
    Returns:
        - List string like '["Apple", "Tesla"]' if query is valid
        - An error string message if invalid or incomplete query
    """
    template = PromptTemplate(
        template="""You are a helpful assistant that extracts valid public company names from a user's query and determines if the query is related to comparing stock prices.

User Query: {query}

Instructions:
- If the query contains at least 2 valid, well-known company names, respond with just a list like: ["Apple", "Coca Cola"]
- If it contains only 1 company name, respond with: "Please enter at least 2 company names to compare their stock prices."
- If it contains no valid company names, respond with: "Invalid company name. Please provide valid company names such as 'Apple', 'Google', 'Microsoft', 'Tesla'."
- If the query is unrelated to stock comparison (like music, jokes, movies, etc), respond with: "I'm an agent that compares stock prices between companies. Please enter valid company names."
- If even a single invalid company name is found then where Invalid is written place the invalid company name and respond with: "Please enter valid company name. Invalid Company is not a valid company." 

Respond only with the appropriate message. Do not explain what you're doing.
""",
        input_variables=["query"],
    )

    prompt = template.format(query=user_query)
    response = model.invoke(prompt)
    return response.content.strip()


def stock_query(user_query):
    # Step 1: Extract string response from model
    raw_response = extract_valid_companies(user_query)

    # Step 2: Try parsing response into a list
    company_list = None
    try:
        if raw_response.startswith("[") and raw_response.endswith("]"):
            parsed = ast.literal_eval(raw_response)
            if isinstance(parsed, list):
                company_list = [c.strip() for c in parsed]  # Clean company names
    except Exception as e:
        print(f"[QueryParser Error] {e}")

    # Step 3: If parsing was successful, use the ticker tool
    if company_list:
        print(f"\n Valid Companies Found: {company_list}")
        company_string = ", ".join(company_list)

        ticker_tool = TickerLookupTool()
        ticker_result = ticker_tool._run(company_string)

        print("\n Ticker Result:", ticker_result)

        # Step 3: If ticker_result is valid, generate report
        if "Invalid company names" not in ticker_result:
            tickers = [t.strip() for t in ticker_result.split(",")]
            report = generate_full_report(tickers)
            print("\n Final Investment Report:\n")
            print(report)
        else:
            print("\n Ticker lookup failed. Please re-enter with valid company names.")


    else:
        # If parsing failed, just print the raw response
        print("\n Invalid query or not enough valid company names.")
        print("🪵 Raw LLM Response:", raw_response)




if __name__ == "__main__":
    user_query = input("Enter your investment query: ")
    main(user_query)






# class TickerLookupTool:
#     def _run(self, company_string: str) -> str:
#         # Dummy lookup - you should replace this with your real implementation
#         valid = {
#             "Apple": "AAPL",
#             "Tesla": "TSLA",
#             "Google": "GOOGL",
#             "Microsoft": "MSFT",
#             "Coca Cola": "KO"
#         }
#         tickers = []
#         for name in company_string.split(","):
#             name = name.strip()
#             if name in valid:
#                 tickers.append(valid[name])
#             else:
#                 return f"Invalid company names: {name}"
#         return ", ".join(tickers)


# def generate_full_report(tickers: list) -> str:
#     # Dummy report generator
#     report = "\n".join([f"- {ticker}: Sample report data." for ticker in tickers])
#     return report

# # ----------------------------- Main Logic ----------------------------- #

# def main(user_query):
#     raw_response = extract_valid_companies(user_query)

#     company_list = None
#     try:
#         if raw_response.startswith("[") and raw_response.endswith("]"):
#             parsed = ast.literal_eval(raw_response)
#             if isinstance(parsed, list):
#                 company_list = [c.strip() for c in parsed]
#     except Exception as e:
#         print(f"[QueryParser Error] {e}")

#     if company_list:
#         print(f"\n Valid Companies Found: {company_list}")
#         company_string = ", ".join(company_list)

#         ticker_tool = TickerLookupTool()
#         ticker_result = ticker_tool._run(company_string)

#         print("\n Ticker Result:", ticker_result)

#         if "Invalid company names" not in ticker_result:
#             tickers = [t.strip() for t in ticker_result.split(",")]
#             report = generate_full_report(tickers)
#             print("\n📘 Final Investment Report:\n")
#             print(report)
#         else:
#             print("\n Ticker lookup failed. Please re-enter with valid company names.")

#     else:
#         print("\n Invalid query or not enough valid company names.")
#         print("🪵 Raw LLM Response:", raw_response)

# # ----------------------------- Entry Point ----------------------------- #
# if __name__ == "__main__":
#   user_query = input("Enter your investment query: ")
#   main(user_query)

