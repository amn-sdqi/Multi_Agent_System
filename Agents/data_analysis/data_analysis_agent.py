import pandas as pd
import numpy as np
import os
import json
import re
import shutil
from typing import Any, Dict, List
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import subprocess
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datetime import datetime

# Configuration
MAX_DISPLAY_ROWS = 20
MAX_OUTPUT_LENGTH = 1000

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Setup output directory
if os.path.exists("output"):
    shutil.rmtree("output")
os.makedirs("output", exist_ok=True)

# Global variables
global_df = None
schema = {}
last_analysis_result = None

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset with encoding detection"""
    if path.endswith(".csv"):
        encodings = ["utf-8", "latin1", "iso-8859-1", "utf-16"]
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode {path} with common encodings.")
    elif path.endswith(".json"):
        return pd.read_json(path)
    else:
        raise ValueError("Unsupported file type")

# Initialize the LLM
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY
)

def get_llm():
    """Return the LLM instance."""
    return llm

def llm_call(prompt: str, operation_name: str = "") -> str:
    """Directly call the LLM and return the response."""
    response = llm.invoke(prompt)
    return response.content.strip()


def generate_schema(df: pd.DataFrame) -> str:
    """Generate schema information"""
    return json.dumps({
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }, indent=2)

def classify_query_complexity(query: str) -> str:
    """Classify query complexity"""
    routing_prompt = f"""
Classify this query as either "simple" or "complex":
- simple: Basic display, head, tail, describe, shape, columns
- complex: Analysis requiring insights, grouping, correlations

Query: "{query}"
Return ONLY: simple or complex
"""
    
    result = llm_call(routing_prompt, "Query Classification")

    if "failed" in result or "unavailable" in result or "exceeded" in result:
        simple_keywords = ['head', 'tail', 'info', 'describe', 'shape', 'show', 'display', 'first', 'last']
        return "simple" if any(keyword in query.lower() for keyword in simple_keywords) else "complex"
    
    return result.lower().strip()

def execute_code(code: str, df: pd.DataFrame) -> Any:
    """Safe code execution"""
    dangerous_patterns = ['__', 'os.system', 'subprocess', 'eval(', 'exec(', 'open(', 'input(', 'sys.exit']
    if any(pattern in code.lower() for pattern in dangerous_patterns):
        return "Unsafe code detected"
    
    try:
        safe_builtins = {
            'print': print, 'len': len, 'range': range,
            'min': min, 'max': max, 'sum': sum,
            'sorted': sorted, 'abs': abs, 'round': round,
            'str': str, 'int': int, 'float': float
        }
        context = {'df': df.copy(), 'pd': pd, 'np': np, **safe_builtins}
        
        if '\n' in code or '=' in code:
            exec(code, context)
            result = context.get('df', "Code executed successfully")
        else:
            result = eval(code, context)
        return format_result(result)
    except Exception as e:
        return f"Execution Error: {str(e)}"

def format_result(result: Any) -> str:
    """Format result with automatic truncation"""
    if isinstance(result, pd.DataFrame):
        if len(result) > MAX_DISPLAY_ROWS:
            formatted = result.head(MAX_DISPLAY_ROWS).to_string(index=True)
            return f"DataFrame with {len(result)} rows (showing first {MAX_DISPLAY_ROWS}):\n{formatted}\n... ({len(result) - MAX_DISPLAY_ROWS} more rows)"
        return result.to_string(index=True)
    elif isinstance(result, pd.Series):
        if len(result) > MAX_DISPLAY_ROWS:
            formatted = result.head(MAX_DISPLAY_ROWS).to_string()
            return f"Series with {len(result)} values (showing first {MAX_DISPLAY_ROWS}):\n{formatted}\n... ({len(result) - MAX_DISPLAY_ROWS} more values)"
        return result.to_string()
    
    result_str = str(result)
    if len(result_str) > MAX_OUTPUT_LENGTH:
        return result_str[:MAX_OUTPUT_LENGTH] + f"... (truncated, {len(result_str) - MAX_OUTPUT_LENGTH} more characters)"
    return result_str

def generate_code(query: str, schema: str) -> str:
    """Generate pandas code."""
    prompt = f"""
Convert the natural language query into correct and safe pandas code.

Rules:
1. Use 'df' as the DataFrame variable.
2. Return ONLY valid Python code (no explanations, markdown, or ``` blocks).
3. Do NOT use '.loc' after 'groupby()', 'mean()', or any chained statistical operation.
4. Filter rows using boolean indexing BEFORE applying 'groupby' or aggregations.
5. For numeric column operations, use: df.select_dtypes(include=['number'])
6. Avoid using 'print()' – just return the code lines.

Schema: {schema}
Query: "{query}"

Generate only the pandas code:
"""

    
    try:
        result = llm_call(prompt)
        code = re.sub(r'^```(?:python)?\s*', '', result).rstrip('`')
        return code.replace("print(", "# print(").strip()
    except Exception:
        # Simple fallbacks
        query_lower = query.lower()
        fallbacks = {
            'head': "df.head()",
            'tail': "df.tail()",
            'info': "df.info()",
            'describe': "df.describe()",
            'shape': "df.shape",
            'columns': "df.columns"
        }
        for key, code in fallbacks.items():
            if key in query_lower:
                return code
        return "df.head()"

def generate_viz_code(query: str, schema: str) -> str:
    """Generate visualization code"""
    prompt = f"""
Create matplotlib/seaborn visualization code. Rules:
1. Use 'df' as dataframe
2. No plt.show()
3. Return only code
4. For correlation matrix: use sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True)
5. If the user asks for a subplot, generate a matplotlib subplot (using plt.subplots) instead of a single plot
6. Do not use plt.figure when generating subplot use plt.subplots
7. Always handle numeric data properly
8. When using palette in countplot, always set hue to the same variable as x and use legend=False

Schema: {schema}
Query: "{query}"

Generate visualization code:
"""
    
    result = llm_call(prompt, "Visualization Generation")

    if "failed" in result or "unavailable" in result or "exceeded" in result:
        return "plt.figure(figsize=(10,6))\ndf.hist()\nplt.title('Data Distribution')"
    
    # More comprehensive cleaning
    code = result.strip()
    
    # Remove any text before the first code block
    if '```' in code:
        # Extract everything between the first ``` and last ```
        parts = code.split('```')
        if len(parts) >= 3:
            # Take the middle part (the actual code)
            code = parts[1]
            # Remove language identifier if present
            if code.startswith('python\n'):
                code = code[7:]
        else:
            # Fallback: remove opening backticks only
            code = re.sub(r'^```(?:python)?\s*', '', code)
    
    # Remove any remaining backticks and explanatory text
    code = code.strip('`').strip()
    
    # Remove common explanatory phrases that might slip through
    explanatory_patterns = [
        r'^Here is.*?:\s*',
        r'^The.*?code.*?:\s*',
        r'^```.*?\n',
        r'\n```.*?$'
    ]
    
    for pattern in explanatory_patterns:
        code = re.sub(pattern, '', code, flags=re.IGNORECASE | re.MULTILINE)
    
    return code.strip()

def execute_viz(code: str, df: pd.DataFrame) -> str:
    """Execute visualization"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join("output", f"plot_{timestamp}.png")
    try:
        context = {'df': df.copy(), 'plt': plt, 'sns': sns, 'np': np}
        # print(code)
        exec(code, context)
        plt.tight_layout()
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        return image_path
    except Exception as e:
        plt.close()
        return f"Visualization failed: {str(e)}"

def open_image(image_path: str):
    """Open image based on OS"""
    try:
        if platform.system() == 'Windows':
            os.startfile(image_path)
        elif platform.system() == 'Darwin':
            subprocess.call(['open', image_path])
        else:
            subprocess.call(['xdg-open', image_path])
    except:
        print(f"Image saved at: {os.path.abspath(image_path)}")

def generate_summary(query: str, result: str, code: str = "") -> str:
    """Generate result summary."""
    prompt = f"""
You are a data analyst assistant.

Given the following query, its analysis result (as a table or value), and the code used to get it — your job is to write a 2-3 sentence summary that answers the user's question directly and clearly.

Do NOT explain how the result was calculated. Focus ONLY on summarizing the result.

Query: {query}
Result: {str(result)[:500]}
Code: {code}

Final Summary:
"""
    
    try:
        return llm_call(prompt)
    except Exception:
        return f"Analysis completed for: {query}"

@tool
def smart_analysis_tool(query: str) -> str:
    """
    Performs data analysis operations including basic display, statistics, 
    exploration, and complex analysis. Use for any dataset questions.
    """
    global last_analysis_result
    
    if global_df is None:
        return "No dataset loaded. Please load a dataset first."
    
    complexity = classify_query_complexity(query)
    
    try:
        code = generate_code(query, schema)
        raw_result = execute_code(code, global_df)
        formatted_result = format_result(raw_result)
        
        # Check for execution errors
        if "Error:" in str(raw_result):
            summary = f"Analysis failed: {raw_result}"
        else:
            if complexity == "simple":
                summary = f"{query}"
            else:
                summary = generate_summary(query, formatted_result, code)
        
        # Store result
        last_analysis_result = {
            'query': query,
            'summary': summary,
            'raw_result': formatted_result,
            'code': code,
            'complexity': complexity,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        # Return formatted response
        response = f"**Analysis Summary:**\n{summary}\n\n**Result:**\n{formatted_result}"
        if complexity == "complex":
            response += f"\n\n**Code Used:** {code}"
        
        return response
        
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        last_analysis_result = {
            'query': query,
            'summary': error_msg,
            'raw_result': error_msg,
            'code': 'N/A',
            'complexity': 'error',
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        return error_msg

@tool
def visualization_tool(query: str) -> str:
    """Create data visualizations with AI-generated code."""
    global last_analysis_result
    
    if global_df is None:
        return "No dataset loaded. Please load a dataset first."
    
    try:
        code = generate_viz_code(query, schema)
        image_path = execute_viz(code, global_df)
        
        if "failed" in image_path.lower() or "error" in image_path.lower():
            return f"Visualization failed: {image_path}"
        
        # Generate insights
        try:
            insights = llm_call(f"Describe insights from visualization: {query}\nSchema: {schema}")
        except Exception:
            insights = f"Visualization created for: {query}"
        
        # Store result
        last_analysis_result = {
            'query': query,
            'summary': f"Visualization: {insights}",
            'raw_result': f"Chart saved to: {image_path}",
            'code': code,
            'complexity': 'visualization',
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        # Open image
        open_image(image_path)
        
        return f"**Visualization Created:**\n{insights}\n\n**File:** {image_path}\n\n**Code:** {code}"
        
    except Exception as e:
        error_msg = f"Visualization error: {str(e)}"
        return error_msg

@tool
def suggestion_tool(input_text: str) -> str:
    """Generate valid analysis questions based on dataset schema and types."""
    if not schema:
        return "No dataset loaded. Please load a dataset first to get suggestions."

    prompt = f"""
    You are a smart data analysis assistant.

    Based on the dataset schema below (which includes column names and types), generate exactly 5 or 6 specific and valid **natural language analysis questions** a user might ask.

    Only ask questions that are logically valid for the **column's data type**.

    Rules:
    - For **categorical columns**, you can ask about:
        • Most/least common category
        • Category-wise distribution
        • Count of unique categories
        • Relationships with other categorical or numerical columns
    - For **numerical columns**, you can ask about:
        • Average, min, max, sum, distribution
        • Correlation with other columns
        • Trends over time (if there's a date column)
    - For **date columns**, you can ask about:
        • Trends over time
        • Time-based grouping (daily, monthly, yearly)
    - DO NOT ask about impossible analysis (e.g., "sum of a category")

    Format:
    Return only the questions as a clean numbered list.
    Do NOT include explanations, markdown, or extra characters.

    Schema:
    {schema}

    Format:
    1. Question one
    2. Question two
    ...
    """

    try:
        suggestions = llm_call(prompt)
        return suggestions
    except Exception as e:
        return f"Could not generate suggestions: {str(e)}"

# Initialize tools and agent
tools = [smart_analysis_tool, visualization_tool, suggestion_tool]
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if llm:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
else:
    agent = None

# CLI Interface
@tool
def data_analysis(path: str, user_query: str):
    """
    This Agent is an Data Analysis Agent
    It will analyse the data and can also visualize and return an image
    """

    # Load dataset

    global_df = load_dataset(path)
    schema = generate_schema(global_df)

    suggestion_tool.invoke(user_query)          
   
    
    if agent:
        print(f"\n{suggestion_tool.invoke('')}")

    # Main loop
    while True:
        try:
            user_query = input("\nQuery: ").strip()
            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_query:
                continue

            # Handle special commands
            if user_query.lower() == "suggestions":
                print("\n**Suggested Analyses:**")
                print(suggestion_tool.invoke(''))
                continue

            result = agent.invoke({"input": user_query})
            print(f"\n{result['output']}")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nSession ended!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
