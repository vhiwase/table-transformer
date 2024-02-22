# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/zero_shot_classification.py

import torch
import logging
import transformers
import os
import gc
import torch
import torch.nn.functional as F
from PIL import Image
import torch
import json

import os
import gc
import io

import pathlib

FILE = pathlib.Path(__file__)
ROOT = FILE.parent

from ts.torch_handler.base_handler import BaseHandler
from inference import TableExtractionPipeline

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

class TableTransformerModelHandler(BaseHandler):

    def initialize(self, context):
        """
        Initialize function loads the model and the tokenizer

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model or
            tokenizer is missing
        """

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")

        # use GPU if available
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info(f'Using device {self.device}')

        # load the model
        model_file = self.manifest['model']['modelFile']
        model_path = os.path.join(model_dir, model_file)

        pipe = None

        if os.path.isfile(model_path):
            pipe = TableExtractionPipeline(det_device=self.device,
                                    str_device=self.device,
                                    det_config_path=(ROOT / "detection_config.json").as_posix(),
                                    det_model_path=model_path,
                                    str_config_path=(ROOT / "structure_config.json").as_posix(),
                                    str_model_path=model_path)
            logger.info(f'Successfully loaded model from {model_file} and Pipeline Object at: {pipe}')
        else:
            raise RuntimeError('Missing the model file')

        self.initialized = True
        self.framework = "pt"
        self.pipe = pipe

    def preprocess(self, data):

        dummy_bytearray = data[0].get('body')
        img = Image.open(io.BytesIO(dummy_bytearray))  
        img = img.convert("RGB")

        mode = data[0].get('mode').decode('utf-8')

        objects = data[0].get('objects')
        objects = objects and objects.decode('utf-8')
        try:
            objects = bool(eval(objects))
        except:
            objects = bool(objects)

        cells = data[0].get('cells')
        cells = cells and cells.decode('utf-8')
        try:
            cells = bool(eval(cells))
        except:
            cells = bool(cells)

        html = data[0].get('html')
        html = html and html.decode('utf-8')
        try:
            html = bool(eval(html))
        except:
            html = bool(html)

        csv = data[0].get('csv')
        csv = csv and csv.decode('utf-8')
        try:
            csv = bool(eval(csv))
        except:
            csv = bool(csv)

        crops = False
        tokens = []

        return img, mode, objects, cells, html, csv, crops, tokens

    def inference(self, img, mode, objects, cells, html, csv, crops, tokens):
        # Read image from the binary stream using PIL
        result_dict = {}

        if mode == 'recognize':
            extracted_table = self.pipe.recognize(img, tokens, out_objects=objects, out_cells=cells,
                                out_html=html, out_csv=csv)
            print("Table(s) recognized.")
            result_dict = extracted_table

        if mode == 'detect':
            detected_tables = self.pipe.detect(img, tokens, out_objects=objects, out_crops=crops)
            print("Table(s) detected.")
            result_dict = detected_tables

        return result_dict
    
    def postprocess(self, output: list):
        """
        Write post process logic
        """
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f'Postprocessing successfully computed')
        return [output]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        img, mode, objects, cells, html, csv, crops, tokens = self.preprocess(data)
        logger.info(f"img{img}, mode:{mode}, objects:{objects}, cells:{cells}, html:{html}, csv:{csv}, crops:{crops}, tokens:{tokens}")
        model_out = self.inference(img, mode, objects, cells, html, csv, crops, tokens)
        return self.postprocess(model_out)

_table_transformer_model_service = TableTransformerModelHandler()

def table_transformer_model_handler(data, context):
    if not _table_transformer_model_service.initialized:
        _table_transformer_model_service.initialize(context)

    if data is None:
        return None

    return _table_transformer_model_service.handle(data, context)