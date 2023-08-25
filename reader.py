# Story Reader - by Sean McManus - www.sean.co.uk
import os
import pygame
import pyttsx3
from datetime import datetime

stories = os.listdir("stories")
path = "stories/" + stories.sort()[-1] # grab the latest story

win_width, win_height = 720, 720
pygame.init()
windowSurface = pygame.display.set_mode((win_width, win_height))
pygame.mouse.set_visible(False)
voice = pyttsx3.init()
voice.setProperty('rate', 170)

story_files, image_files = [], []
for file in os.listdir(path):
    if file.lower().endswith('.jpg'):
        image_files.append(file)
    elif file.lower().startswith('story'):
        story_files.append(file)
story_has_title = len(story_files) > len (image_files)
story_files = sorted(story_files)
image_files = sorted(image_files)

for number, story in enumerate(story_files):
    if story_has_title:
        image_path = os.path.join(path, image_files[max(0, number - 1)])
    else:
        image_path = os.path.join(path, image_files[number])
    image_to_show = pygame.image.load(image_path)
    image_to_show = pygame.transform.scale(image_to_show, (win_width, win_height))
    windowSurface.blit(image_to_show, (0,0))
    pygame.display.update()
    story_path = os.path.join(path, story)
    with open(story_path, "r") as file:
        story = file.readlines()
    voice.say(story)
    voice.runAndWait()
pygame.quit()