import os
from typing import Any
import logging

import click
import tqdm

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import yaml
from DetPlayground.Models import MODEL_COLLECTION
from DetPlayground.Dataset.CocoDataset import CocoDataset
from DetPlayground.Loss.YoloXLossCalculator import YoloXLossCalculator

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
    model = MODEL_COLLECTION[args["model"]["type"]]
    model = model(args["model"])

    # Dataloader
    dataset = CocoDataset(args=args["training"]["data"])
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=args["training"]["data"]["batchSize"], 
    )
    
    # Loss
    loss_calculator = YoloXLossCalculator(args["model"])
    
    # Optimizer and Scheduler
    
    
    model.train()
    device = torch.device("cuda:0")
    model.to(device=device)
    for data, target in tqdm.tqdm(dataloader):
        data = data.to(device=device)
        target = target.to(device=device)
        output = model(data)
        
        # loss, reg_weight * loss_iou, loss_obj, loss_cls, num_fore_ground / max(num_groud_truth, 1)
        losses = loss_calculator.calculate_losses(
            outputs = output, 
            targets = target
        )
        
        model.parameters()

if __name__ == "__main__":
    main()
