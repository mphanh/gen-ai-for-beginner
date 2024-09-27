import openai
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the API key, base URL, and version for Azure OpenAI
openai.api_key = os.environ["AZURE_OPENAI_KEY"]
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]  # Example: "https://your-openai-resource-name.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = os.environ["AZURE_OPENAI_VERSION"]  # Example: "2023-03-15-preview"

# Set your deployment (model) ID
deployment = os.environ['AZURE_OPENAI_DEPLOYMENT']  # Example: "gpt-35-turbo"

no_recipe = input("No of recipes (for example, 5): ")

ingredients = input("List of ingredients (for example, milk, eggs, flour): ")

filter = input("Filter (for example, vegan, gluten free, no peanut): ")
# Define the prompt and messages
prompt = f"Show me {no_recipe} recipes with {ingredients} that are {filter}. Per receipe, list all ingredients used: "
messages = [
    # {"role": "system", "content": "You are a curator at a museum."},
    {"role": "user", "content": prompt}
]

# Use the latest version's completion function to generate a response
completion=openai.chat.completions.create(model=deployment, messages=messages, max_tokens=600, temperature=0.1)

# Print the response
print("Recipes: ")
print(completion.choices[0].message.content)

old_promt_result = completion.choices[0].message.content
prompt_shopping = "Produce a shopping list excluding ingredients I already have at home: "
new_promt = f"Given ingredients at home {ingredients} and these generated recipes: {old_promt_result}, {prompt_shopping} "
messages = [{"role": "user", "content": new_promt}]
completion=openai.chat.completions.create(model=deployment, messages=messages, max_tokens=600, temperature=0.1)

print("Shopping list: ")
print(completion.choices[0].message.content)