#!/usr/bin/env python
import wandb
import json

# 1. Start a W&B run
wandb.init(project='undeepwave', entity='siggyf')

# 2. Save model inputs and hyperparameters
config = wandb.config
with open('parameters.json') as f:
    parameters = json.load(f)
    config.update(parameters)

wandb.log({"result": 1})

