import openai
import json
from openai import OpenAI
import yfinance as yf
from datetime import datetime
from duckduckgo_search import DDGS
import time  # Add this at the top with other imports

# OpenAI API key is read from the system environment
# Ensure you have set it using: set OPENAI_API_KEY=your_api_key
client = OpenAI()  # This will automatically use your environment variable

# Define the tools
def calculate(expression):
    """Evaluate a mathematical expression"""
    try:
        # Attempt to evaluate the math expression
        result = eval(expression)
        return result  # Return the numeric result directly
    except Exception as e:
        return f"Error: Invalid expression - {str(e)}"

# Add stock price lookup function
def get_stock_price(query):
    """Get current stock price for a given ticker or company name"""
    try:
        # If query might be a company name, try to get the ticker first
        if not query.isalpha() or len(query) > 5:
            # This is a simplified approach. In a real application, you might want
            # to use a more robust company name to ticker mapping
            company_to_ticker = {
                "microsoft": "MSFT",
                "apple": "AAPL",
                "google": "GOOGL",
                "amazon": "AMZN",
                "meta": "META",
                # Add more mappings as needed
            }
            query = company_to_ticker.get(query.lower(), query)

        # Get stock info
        stock = yf.Ticker(query)
        info = stock.info
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        company_name = info.get('longName', query)
        
        if current_price is None:
            return f"Error: Could not find price for {query}"
            
        return {
            "symbol": query.upper(),
            "company_name": company_name,
            "current_price": current_price,
            "currency": info.get('currency', 'USD'),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return f"Error looking up stock price: {str(e)}"

def web_search(query, max_results=3):
    """Search the web using DuckDuckGo for stock analysis and investment opinions"""
    try:
        # Modify query to focus on analyst opinions and stock analysis
        refined_query = f"{query} stock analyst ratings investment opinion site:seekingalpha.com OR site:marketwatch.com OR site:finance.yahoo.com"
        
        with DDGS() as ddgs:
            time.sleep(1)
            results = list(ddgs.text(
                refined_query,
                max_results=max_results,
                region='wt-wt',
                safesearch='off',
                time='m'  # Get results from last month
            ))
            
        if not results:
            return "No recent analyst opinions found."
            
        formatted_results = []
        for r in results:
            formatted_results.append({
                "title": r.get('title', ''),
                "link": r.get('link', ''),
                "snippet": r.get('snippet', r.get('body', '')),
                "date": r.get('date', '')
            })
        return formatted_results
    except Exception as e:
        if "Ratelimit" in str(e):
            return "Search temporarily unavailable due to rate limiting. Please try again in a few moments."
        return f"Error performing search: {str(e)}"

# Define the metadata for the functions
functions = [
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression. Can handle basic arithmetic operations (+, -, *, /).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given stock symbol or company name",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Stock symbol (e.g., MSFT) or company name (e.g., Microsoft)",
                }
            },
            "required": ["query"],
        }
    },
    {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo for general information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }
]

# Function to process chained tool calls
def process_chained_operations(initial_response, messages):
    result = None
    
    while initial_response.choices[0].finish_reason == "function_call":
        function_call = initial_response.choices[0].message.function_call
        function_name = function_call.name
        arguments = json.loads(function_call.arguments)
        
        # Execute the appropriate function
        if function_name == "calculate":
            result = calculate(arguments['expression'])
        elif function_name == "get_stock_price":
            result = get_stock_price(arguments['query'])
        elif function_name == "web_search":
            max_results = arguments.get('max_results', 3)
            result = web_search(arguments['query'], max_results)
        
        print(f"After {function_name}: {result}")
        
        # Add the function result to messages
        messages.append({
            "role": "function",
            "name": function_name,
            "content": str(result)
        })
        
        # Add a message to guide the next step
        messages.append({
            "role": "assistant",
            "content": f"I got the result: {result}. Let me continue with the next operation if needed."
        })
        
        # Get the next function call if needed
        initial_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=functions,
        )
    
    # After all function calls are complete, get final natural language response
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    return final_response.choices[0].message.content

# Main function to run the tool
def main():
    user_input = input("Enter your query: ")

    messages = [
        {
            "role": "system", 
            "content": """You are a helpful assistant that can:
            1. Look up stock prices using the get_stock_price function
            2. Evaluate mathematical expressions using the calculate function
            3. Search the web using the web_search function
            
            When handling queries:
            - For stock prices, use get_stock_price
            - For news about a company, use web_search only if specifically requested
            - For calculations, use the calculate function
            - Combine tools only when explicitly asked by the user
            
            Examples:
            - "What's AAPL stock price?" → Use get_stock_price only
            - "What's AAPL stock price and latest news?" → Use both get_stock_price and web_search
            - "Calculate AAPL stock value in HKD" → Use get_stock_price then calculate
            
            Keep responses concise but informative."""
        },
        {
            "role": "user", 
            "content": user_input
        }
    ]

    # Call the OpenAI ChatCompletion API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
    )

    # Process and display the result
    final_result = process_chained_operations(response, messages)
    print(final_result)

# Run the script
if __name__ == "__main__":
    main()
