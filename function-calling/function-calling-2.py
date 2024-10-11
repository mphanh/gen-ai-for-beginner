import openai
import json
import os
from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI service details
azure_openai_api_base = os.environ["AZURE_OPENAI_ENDPOINT"]  # Replace with your resource name
azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]  # Replace with your API key
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT"]  # Replace with your deployment name, e.g., "gpt-4-deployment"

# Configure OpenAI with Azure credentials
openai.api_type = "azure"
openai.api_base = azure_openai_api_base
openai.api_version = "2023-06-01-preview"  # Use the appropriate API version
openai.api_key = azure_openai_api_key

def get_weather(location):
    """
    Sample function to get weather information for a given location.
    In a real-world scenario, this function would integrate with a weather API.
    """
    # Simulated weather data
    weather_data = {
        "New York": {"temperature": "15°C", "description": "Sunny"},
        "London": {"temperature": "10°C", "description": "Cloudy"},
        "Tokyo": {"temperature": "20°C", "description": "Rainy"},
    }
    
    weather = weather_data.get(location, {"temperature": "N/A", "description": "Location not found"})
    return f"The current weather in {location} is {weather['description']} with a temperature of {weather['temperature']}."

# Define the functions that the model can call
functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather information for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., New York, NY"
                }
            },
            "required": ["location"]
        }
    }
]

# Define the initial conversation
messages = [
    {"role": "system", "content": "You are a helpful assistant that can provide weather information."},
    {"role": "user", "content": "Can you tell me the weather in Tokyo today?"}
]

# Make the initial API call with function definitions
response = openai.chat.completions.create(
    model=deployment_name,  # e.g., "gpt-4-deployment"
    messages=messages,
    functions=functions,
    function_call="auto"  # Let the model decide to call a function
)

# Extract the first choice from the response
choice = response.choices[0]

# Check if the model chose to call a function
if 'function_call' in choice.message:
    function_name = choice.message.function_call.name
    arguments = json.loads(choice.message.function_call.arguments)
    
    if function_name == "get_weather":
        location = arguments.get("location")
        # Execute the function
        function_response = get_weather(location)
        
        # Append the function call and its response to the messages
        messages.append(choice.message)  # The assistant's message with function_call
        messages.append({
            "role": "function",
            "name": "get_weather",
            "content": function_response
        })
        
        # Make a follow-up call to get the assistant's response incorporating the function's output
        final_response = openai.chat.completions.create(
            model=deployment_name,
            messages=messages
        )
        
        # Print the assistant's reply
        print(final_response.choices[0].message.content)
else:
    # If no function was called, print the assistant's message
    print(choice.message)
