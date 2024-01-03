import pathlib
import os
import shutil
import random
        
# Define the split ratios
train_ratio = 0.7  # 70% for training
test_ratio = 0.2   # 20% for testing
val_ratio = 0.1    # 10% for validation

def generate_data(path):
    if path is None:
        return
    else:
        print(f"Processing {path}")
    for folder in os.listdir(path):
        folder_path = path / folder
        parts = list(folder_path.parts)
        if 'label_studio' in parts:
            parts[parts.index('label_studio')] = 'tabel_transformer'
            output_path = pathlib.Path(*parts)
        else:
            output_path = None
            continue
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)
        
        input_image_path = folder_path / 'images'
        input_file_path = folder_path / 'Annotations'
        
        output_image_path = output_path  / 'images'
        output_train_path = output_path / 'train'
        output_test_path = output_path / 'test'
        output_val_path = output_path / 'val'
        
        os.makedirs(output_image_path, exist_ok=True)
        os.makedirs(output_train_path, exist_ok=True)
        os.makedirs(output_test_path, exist_ok=True)
        os.makedirs(output_val_path, exist_ok=True)
        
        file_list = os.listdir(input_file_path)
        random.shuffle(file_list)
        
        total_files = len(file_list)
        train_split = int(train_ratio * total_files)
        test_split = int(test_ratio * total_files)
        
        # Split files into train, test, and validation sets
        train_files = file_list[:train_split]

        test_files = file_list[train_split:train_split + test_split]

        val_files = file_list[train_split + test_split:]

        for img_name in os.listdir(input_image_path):
            source = input_image_path/img_name
            target = output_image_path/img_name.replace('.png', '.jpg')
            shutil.copy(source, target)
    
        with open(output_path/'images_filelist.txt', 'a') as f:
            for image_file in os.listdir(output_image_path):
                text = 'images/'+image_file+'\n'
                f.write(text)

        for file_name in train_files:
            source = input_file_path/file_name
            target = output_train_path/file_name
            shutil.copy(source, target)

        with open(output_path/'train_filelist.txt', 'a') as f:
            for train_file in train_files:
                text = 'train/'+train_file+'\n'
                f.write(text)

        for file_name in test_files:
            source = input_file_path/file_name
            target = output_test_path/file_name
            shutil.copy(source, target)

        with open(output_path/'test_filelist.txt', 'a') as f:
            for test_file in test_files:
                text = 'test/'+test_file+'\n'
                f.write(text)

        for file_name in val_files:
            source = input_file_path/file_name
            target = output_val_path/file_name
            shutil.copy(source, target)

        with open(output_path/'val_filelist.txt', 'a') as f:
            for val_file in val_files:
                text = 'test/'+val_file+'\n'
                f.write(text)


if __name__ == '__main__':
    try:
        FILE_PATH = pathlib.Path(__file__)
    except NameError:
        FILE_PATH = pathlib.Path('.data_generator.py')

    ROOT_PATH = FILE_PATH.parent

    DETECTION_PATH = ROOT_PATH / 'detection'
    STRUCTURE_PATH = ROOT_PATH / 'structure'

    if DETECTION_PATH.is_dir():
        DETECTION_LABEL_STUDIO_PATH = DETECTION_PATH / 'label_studio'
    else:
        DETECTION_LABEL_STUDIO_PATH = None
        
    if STRUCTURE_PATH.is_dir():
        STRUCTURE_LABEL_STUDIO_PATH = STRUCTURE_PATH / 'label_studio'
    else:
        STRUCTURE_LABEL_STUDIO_PATH = None
    path = DETECTION_LABEL_STUDIO_PATH
    generate_data(path)
    path = STRUCTURE_LABEL_STUDIO_PATH
    generate_data(path)

