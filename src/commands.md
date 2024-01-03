# Instruction
Go to src folder path ```table-transformer/src```, open terminal run the following command

# Create and Activate environment
```sh
conda env create -f environment.yml
conda activate tables-detr
```

# Training data creation from label studio exported files
```sh
python "./label_studio_extraction/data_generator.py"
```

# Model Training
```sh
python main.py --data_type detection --config_file detection_config.json --data_root_dir "./label_studio_extraction/detection/tabel_transformer/border" --epochs 500 --checkpoint_freq 500 --device cuda --model_save_dir "./model_save_dir" 
```
```sh
python main.py --data_type detection --config_file detection_config.json --data_root_dir "./label_studio_extraction/detection/tabel_transformer/borderless" --epochs 500 --checkpoint_freq 500 --device cuda --model_save_dir "./model_save_dir" 
```
```sh
python main.py --data_type structure --config_file structure_config.json --data_root_dir "./label_studio_extraction/structure/tabel_transformer/border" --epochs 500 --checkpoint_freq 500 --device cuda --model_save_dir "./model_save_dir" 
```
```sh
python main.py --data_type structure --config_file structure_config.json --data_root_dir "./label_studio_extraction/structure/tabel_transformer/borderless" --epochs 500 --checkpoint_freq 500 --device cuda --model_save_dir "./model_save_dir" 
```

# Model Prediction
```sh
python inference.py --image_dir "./label_studio_extraction/detection/tabel_transformer/border/images" --mode detect --detection_config_path detection_config.json --detection_model_path "./model_save_dir/model.pth" --detection_device cpu --out_dir "./label_studio_extraction/detection/tabel_transformer/border/prediction" -c -v -p -o -l -m -z
```
```sh
python inference.py --image_dir "./label_studio_extraction/detection/tabel_transformer/borderless/images" --mode detect --detection_config_path detection_config.json --detection_model_path "./model_save_dir/model.pth" --detection_device cpu --out_dir "./label_studio_extraction/detection/tabel_transformer/borderless/prediction" -c -v -p -o -l -m -z
```
```sh
python inference.py --image_dir "./label_studio_extraction/structure/tabel_transformer/border/images" --mode recognize --structure_config_path structure_config.json --structure_model_path "./model_save_dir/model.pth" --structure_device cpu --out_dir "./label_studio_extraction/structure/tabel_transformer/border/prediction" -c -v -p -o -l -m -z
```
```sh
python inference.py --image_dir "./label_studio_extraction/structure/tabel_transformer/borderless/images" --mode recognize --structure_config_path structure_config.json --structure_model_path "./model_save_dir/model.pth" --structure_device cpu --out_dir "./label_studio_extraction/structure/tabel_transformer/borderless/prediction" -c -v -p -o -l -m -z
```

# Model Fine Tuning
```sh
python main.py --data_type detection --config_file detection_config.json --data_root_dir "./label_studio_extraction/detection/tabel_transformer/border" --epochs 1 --checkpoint_freq 1 --device cpu --model_save_dir "./model_save_dir" 
```

```sh
python main.py --data_type detection --config_file detection_config.json --data_root_dir "./label_studio_extraction/detection/tabel_transformer/border" --epochs 5 --checkpoint_freq 10 --device cpu --model_save_dir "./fine_tuned_dir" --model_load_path "./model_save_dir/model_1.pth" --load_weights_only
```