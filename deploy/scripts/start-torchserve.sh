#!/bin/bash

# --ncs means the snapshot feature is disabled.

echo "server is starting..."
torchserve --foreground --ncs --model-store model_store --ts-config config.properties