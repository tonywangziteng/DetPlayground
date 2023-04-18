import os
from typing import Any
import logging

import click
import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import yaml
from Models import model_collection
from Dataset.CocoDataset import CocoDataset
from Loss.LossCalculator import LossCalculator

@click.command()
@click.option('--config_name',type = str,  default = "YoloxL.yaml", help='Number of greetings.')
def main(config_name: str):
    # config logging basic configuration
    logging.basicConfig(level=logging.INFO)
    
    config_path: str = os.path.join("../Config", config_name)
    with open(config_path) as f:
        args = yaml.load(f, Loader=yaml.SafeLoader)
        
    os.environ["CUDA_VISIBLE_DEVICE"] = str(args["training"]["GPUIdx"])
        
    # Model preparation
    model = model_collection[args["model"]["type"]]
    model = model(args["model"])

    # Dataloader
    dataset = CocoDataset(args=args["training"]["data"])
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args["training"]["batchSize"], 
    )
    
    loss_calculator = LossCalculator(
        num_classes=args["model"]["num_classes"], 
        strides=model.strides
    )
    
    model.train()
    device = torch.device("cuda:0")
    model.to(device=device)
    for data, target in tqdm.tqdm(dataloader):
        data = data.to(device=device)
        target = target.to(device=device)
        output = model(data)
        
        # losses = loss_calculator.calculate_losses(
        #     outputs = output, 
        #     targets = target
        # )

if __name__ == "__main__":
    main()
