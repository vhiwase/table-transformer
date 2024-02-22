#!/bin/bash

rm -rf model_store
mkdir -p model_store

torch-model-archiver \
--model-name layoutlmv3-for-tokenclassification \
--version 1.0 \
--model-file my_model/model.safetensors \
--handler layoutlmv3_tokenclassification_handler:layoutlmv3_tokenclassification_model_handler \
--extra-files "my_model/config.json,my_model/merges.txt,my_model/optimizer.pt,my_model/preprocessor_config.json,my_model/rng_state.pth,my_model/scheduler.pt,my_model/special_tokens_map.json,my_model/tokenizer_config.json,my_model/tokenizer.json,my_model/trainer_state.json,my_model/training_args.bin,my_model/vocab.json" \
--export-path model_store

echo "Archive created!"