```sh
torch-model-archiver --model-name table_transformer_detection --version 1.0 --model-file my_model/pubtables1m_detection_detr_r18.pth --handler inference_handler:table_transformer_model_handler --extra-files "detection_config.json,structure_config.json,inference.py,postprocess.py,model.py,inference_handler.py,inference.py" --export-path model_store
```

```sh
torch-model-archiver --model-name table_transformer_structure --version 1.0 --model-file my_model/pubtables1m_structure_detr_r18.pth --handler inference_handler:table_transformer_model_handler --extra-files "detection_config.json,structure_config.json,inference.py,postprocess.py,model.py,inference_handler.py,inference.py" --export-path model_store
```