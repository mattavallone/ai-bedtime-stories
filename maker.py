# Story Maker - by Sean McManus - www.sean.co.uk
# Modified by Matt Avallone
import os
import openai
import requests
import datetime
import random
import time
from dotenv import load_dotenv

# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers import DiffusionPipeline
import torch

# model_id = "stabilityai/stable-diffusion-2"

# # Use the Euler scheduler here instead
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 10
high_noise_frac = 0.8


load_dotenv()
OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')
DEEP_AI_KEY = os.getenv('DEEPAI_API_KEY')

def write_list(list_to_write, filename):
    for number, paragraph in enumerate(list_to_write):
        filepath = os.path.join(directory, f"{filename}-{number}.txt")
        with open(filepath, "w") as file:
            file.write(paragraph)

date_time_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
directory = "stories/" + f"Story {date_time_str}"
os.mkdir(directory)
image_prompts, story_paragraphs = [], []
character1_name = input("What is your main character's name? ")
character1_type = input("What kind of a character is that? ")
character2_name = input("What is your second character's name? ")
character2_type = input("And what kind of a character is that? ")
venue = input("Where does the story take place? (e.g. in a castle, on Mars) ")
genre = input("What is your story genre? ")
story_prompt = f"Please write me a short {genre}\
story. In this story, {character1_name} is a\
{character1_type} and {character2_name} is a\
{character2_type}. The story takes place {venue}.\
For each paragraph, write me an image prompt for\
an AI image generator. Each image prompt must\
start in a new paragraph and have the words 'Image\
Prompt:' at the start. Choose a book illustrator\
and put something in the image prompts to say the\
images should be made in the style of that artist."

while len(image_prompts) == 0:
    print("Making ChatGPT request")
    openai.api_key = OPEN_AI_KEY
    ChatGPT_output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a children's author."},
            {"role": "user", "content": story_prompt}
          ] )
    new_story = ChatGPT_output.choices[0].message["content"]
    print(new_story)
    for paragraph in new_story.split("\n\n"):
        if paragraph.startswith("Image Prompt"):
            image_prompts.append(paragraph)
        else:
            story_paragraphs.append(paragraph)
write_list(story_paragraphs, "story")
write_list(image_prompts, "prompt")

for number, image_prompt in enumerate(image_prompts):
    image_prompt += f"{character1_name} is {character1_type} and {character2_name} is {character2_type}. They are {venue}."
    print(f"Generating image {number}")

    prompt = image_prompt
    # image = pipe(prompt).images[0]
    # run both experts
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    filename = f"{number}.png"
    filepath = os.path.join(directory, filename)
    print(f"Image {number} is at {filepath}. Saving now...\n\n")
    image.save(filepath)

print(f"Your files are in {directory}.")