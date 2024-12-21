import ollama
import yfinance as yf

prompt = "What is the stock price of Apple?"

# define a function to use as tool
def get_stock_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    return stock.history(period='1d')['Close'].iloc[-1]

# Pass the function as a tool to Ollama
response = ollama.chat(
    'llama3.2',
    messages=[{'role': 'user', 'content': 'What is the stock price of Apple?'}],
    tools=[get_stock_price],  # Actual function reference
)

# Call the function from the model response
available_functions = {
    'get_stock_price': get_stock_price,
}

for tool in response.message.tool_calls or []:
    function_to_call = available_functions.get(tool.function.name)
    if function_to_call:
        print('Agruments:', tool.function.arguments)
        print('Function output:', function_to_call(**tool.function.arguments))
    else:
        print('Function not found:', tool.function.name)