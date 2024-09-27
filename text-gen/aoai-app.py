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

# Define the prompt and messages
prompt = "Complete the following: Once upon a time there was a"
messages = [
    # {"role": "system", "content": "You are a curator at a museum."},
    {"role": "user", "content": prompt}
]

# Use the latest version's completion function to generate a response
completion=openai.chat.completions.create(model=deployment, messages=messages)
# response = openai.completions.create(
#     deployment_id=deployment,  # Use the deployment ID (e.g., "gpt-35-turbo")
#     messages=messages
# )

# Print the response
print(completion.choices[0].message.content)
