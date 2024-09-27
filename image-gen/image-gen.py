import openai
import os
import requests
from PIL import Image
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Get endpoint and key from environment variables
openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
openai.api_key = os.environ['AZURE_OPENAI_API_KEY']

# Assign the API version (DALL-E is currently supported for the 2023-06-01-preview API version only)
openai.api_version = os.environ["AZURE_OPENAI_VERSION"]
openai.api_type = 'azure'

# Create an image using the image generator API (DALL-E)
def generate_response(prompt):
    response = openai.images.generate(
        model=os.environ['AZURE_OPENAI_DEPLOYMENT'], #deplyment name, not model name
        # image=open("base_image.png", "rb"), #base image for modification or creating a variation
        # mask=open("mask.png", "rb"), #mask that adds to the base image
        prompt=prompt,
        size="1024x1024",
        # response_format="b64_json",
        n=1, #number of images generated
        # temperature=0, #the randomness of the output
    )
    # image = response['data'][0]['b64_json']
    return response

try:
    generation_response = generate_response("Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils")

    image_dir = os.path.join(os.curdir, 'images')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path (note the filetype should be png)
    image_path = os.path.join(image_dir, 'generated-image.png')

    # Retrieve the generated image
    image_url = generation_response.data[0].url  # extract image URL from response
    generated_image = requests.get(image_url).content  # download the image
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    # Display the image in the default image viewer
    image = Image.open(image_path)
    image.show()
except openai.OpenAIError as err:
    print(f"OpenAI API error: {err}")
except requests.exceptions.RequestException as req_err:
    print(f"Request error: {req_err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")