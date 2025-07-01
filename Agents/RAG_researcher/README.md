# AI Agent - Research Assistant


Information Retrieval Agent (Research Assistant)

Task: Build an agent that retrieves and summarizes information from the web or a provided knowledge base (e.g., Wikipedia, PDFs, or web search).

Features:
- Uses LangChain’s web search tool (e.g., Tavily, SerpAPI) or document loader.
- Summarizes findings in concise, user-friendly responses.
- Example query: “Summarize recent advancements in quantum computing.”
- Skills Learned: Web scraping, document loading, summarization, tool integration.
- Tools: LangChain’s WebBaseLoader, TavilySearchResults, or Wikipedia loader.


---

## Tools Used

- LangChain
- Google Gemini (Generative AI)
- DuckDuckGo + Wikipedia tools
- PyPDFLoader (for reading PDFs)
- FAISS (for vector search)
- HuggingFace Embeddings

---

## How to Run

1. Clone the repository

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate


3. Install the dependencies
 pip install -r requirements.txt


4. Create a .env file and add your key:
  GOOGLE_API_KEY=your_google_key_here

5. Run the app:
  python main.py



Example Prompt
“Summarize recent advancements in quantum computing”
or
“Summarize this document: Demo.pdf”


🙋‍♂️ Author
Wajid Khan
GitHub Profile