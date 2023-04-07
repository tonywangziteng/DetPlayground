from typing import Any

import click

import torch
import pycocotools

import yaml
from Models import model_collection
import Models

# @click.command()
# @click.option('--count',type = int,  default=1, help='Number of greetings.')
def main():
    model = Models.Darknet.Darknet53()
    print(model)

if __name__ == "__main__":
    main()
