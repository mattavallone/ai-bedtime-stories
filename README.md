# AI Generated Bedtime Stories

This project uses gpt-3.5-turbo and stable-diffusion-xl-base-1.0 & stable-diffusion-xl-refiner-1.0 to generate children's bedtime stories with illustrations. These stories can then be read aloud for you while showing you the approriate image at each point of the story.

The gpt-3.5-turbo model is used to generate the story text and image prompts. These image prompts get fed into stable-diffusion-xl models to produce the illustrations. The stories can be played using a Raspberry Pi with a display, although this is not required.

Modifications to the original project include using [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) rather than DeepAI's API, which is not free. The model is considerably slower than other options but produces better images.

Based on [Bedtime Stories - create illustrated stories using the ChatGPT and DeepAI (Stable Diffusion) APIs in Python](https://www.sean.co.uk/raspberry_pi/bedtime_stories.shtm)
 
