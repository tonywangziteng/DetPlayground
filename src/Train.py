import os
from typing import Any

import click
import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import yaml
from Models import model_collection
from Dataset.CocoDataset import CocoDataset, coco_collate_fn

@click.command()
@click.option('--config_path',type = str,  default = "../Config/YoloxL.yaml", help='Number of greetings.')
def main(config_path: str):
    with open(config_path) as f:
        args = yaml.load(f, Loader=yaml.SafeLoader)
        
    os.environ["CUDA_VISIBLE_DEVICE"] = str(args["training"]["GPUIdx"])
        
    # Model preparation
    model: nn.Module = model_collection[args["model"]["type"]](args["model"])

    # Dataloader
    dataset = CocoDataset(args=args["training"]["data"])
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args["training"]["batchSize"], 
        collate_fn=coco_collate_fn
    )
    
    model.train()
    device = torch.device("cuda:0")
    model.to(device=device)
    for data, target in tqdm.tqdm(dataloader):
        data = data.to(device=device)
        output = model(data)

if __name__ == "__main__":
    
    main()
