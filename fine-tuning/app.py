import openai
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_VERSION")
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Upload Your Dataset
ft_file = openai.files.create(
  file=open("./training-data.jsonl", "rb"),
  purpose="fine-tune"
)

print(ft_file)
print("Training File ID: " + ft_file.id)

# Create the Fine-tuning job
ft_filejob = openai.fine_tuning.jobs.create(
  training_file=ft_file.id, 
  model="gpt-35-turbo"
)

print(ft_filejob)
print("Fine-tuning Job ID: " + ft_filejob.id)

#  Check the Status of the job
# openai.fine_tuning.jobs.list(limit=<n>) - List the last n fine-tuning jobs
# openai.fine_tuning.jobs.retrieve(<job_id>) - Get details of a specific fine-tuning job
# openai.fine_tuning.jobs.cancel(<job_id>) - Cancel a fine-tuning job
# openai.fine_tuning.jobs.list_events(fine_tuning_job_id=<job_id>, limit=<b>) - List up to n events from the job
# openai.fine_tuning.jobs.create(model="gpt-35-turbo", training_file="your-training-file.jsonl", ...)
# List 10 fine-tuning jobs
openai.fine_tuning.jobs.list(limit=10)

# Retrieve the state of a fine-tune
response=openai.fine_tuning.jobs.retrieve(ft_filejob.id)
print("Job ID:", response.id)
print("Status:", response.status)
print("Trained Tokens:", response.trained_tokens)

# List up to 10 events from a fine-tuning job
openai.fine_tuning.jobs.list_events(fine_tuning_job_id=ft_filejob.id, limit=10)

# Track events to monitor progress
# You can also track progress in a more granular way by checking for events
# Refresh this code till you get the `The job has successfully completed` message
response = openai.fine_tuning.jobs.list_events(ft_filejob.id)

events = response.data
events.reverse()

for event in events:
    print(event.message)

# Retrieve the identity of the fine-tuned model once ready
response = openai.fine_tuning.jobs.retrieve(ft_filejob.id)
fine_tuned_model_id = response.fine_tuned_model
print("Fine-tuned Model ID:", fine_tuned_model_id)
# Test the fine-tuned model
completion = openai.chat.completions.create(
  model=fine_tuned_model_id,
  messages=[
    {"role": "system", "content": "You are Elle, a factual chatbot that answers questions about elements in the periodic table with a limerick"},
    {"role": "user", "content": "Tell me about Strontium"},
  ]
)
print(completion.choices[0].message)
