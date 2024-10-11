import os
import json
import requests
import openai
from dotenv import load_dotenv
load_dotenv()

# Set the API key, base URL, and version for Azure OpenAI
openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]  # Example: "https://your-openai-resource-name.openai.azure.com/"
openai.api_type = "azure"
openai.api_version = os.environ['AZURE_OPENAI_VERSION']  # Example: "2023-03-15-preview"

deployment=os.environ['AZURE_OPENAI_DEPLOYMENT']

messages= [ 
   # {"role": "system", "content": "You are a helpful assistant."}, 
   {"role": "user", "content": "Find me details about a good course for a beginner student to learn Azure with information on the role, product, and level."} ]

def search_courses(role, product, level):
    url = "https://learn.microsoft.com/api/catalog/"
    params = {
        "role": role,
        "product": product,
        "level": level
    }
    response = requests.get(url, params=params)
    modules = response.json()["modules"]
    results = []
    for module in modules[:5]:
        title = module["title"]
        url = module["url"]
        results.append({"title": title, "url": url})
    return str(results)

functions = [
   {
      "name":"search_courses",
      "description":"Retrieves courses from the search index based on the parameters provided",
      "parameters":{
         "type":"object",
         "properties":{
            "role":{
               "type":"string",
               "description":"The role of the learner (i.e. developer, data scientist, student, etc.)"
            },
            "product":{
               "type":"string",
               "description":"The product that the lesson is covering (i.e. Azure, Power BI, etc.)"
            },
            "level":{
               "type":"string",
               "description":"The level of experience the learner has prior to taking the course (i.e. beginner, intermediate, advanced)"
            }
         },
         "required":[
            "role"
         ]
      }
   }
]

response = openai.chat.completions.create(model=deployment, messages=messages, functions=functions, function_call="auto")

print(response.choices[0].message)
response_message = response.choices[0].message

# Check if the model wants to call a function
if response_message.function_call is not None:
   print("Recommended Function call:")
   print(response_message.function_call.name)
   print()

   # Call the function.
   function_name = response_message.function_call.name

   available_functions = {
           "search_courses": search_courses,
   }
   function_to_call = available_functions[function_name]

   function_args = json.loads(response_message.function_call.arguments)
   function_response = function_to_call(**function_args)

   print("Output of function call:")
   print(function_response)
   print(type(function_response))

 # Add the assistant response and function response to the messages
   messages.append( # adding assistant response to messages
      {
         "role": response_message.role,
         "function_call": {
               "name": function_name,
               "arguments": response_message.function_call.arguments,
         },
         "content": None
      }
   )
   messages.append( # adding function response to messages
      {
         "role": "function",
         "name": function_name,
         "content":function_response,
      }
   )

   print("Messages in next request:")
   print(messages)
   print()

   second_response = openai.chat.completions.create(
      messages=messages,
      model=deployment,
      function_call="auto",
      functions=functions,
      temperature=0
   )  # get a new response from GPT where it can see the function response
   print(second_response.choices[0].message)