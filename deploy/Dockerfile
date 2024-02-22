# Builder image
FROM pytorch/torchserve:latest-gpu AS builder

WORKDIR /usr/app
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD inference_handler.py .

RUN rm -rf model_store
RUN mkdir -p model_store
COPY model_store model_store

# CMD ["python", "-c 'while True: pass'"]

RUN torch-model-archiver --model-name table_transformer_detection --version 1.0 --model-file my_model/pubtables1m_detection_detr_r18.pth --handler inference_handler:table_transformer_model_handler --extra-files "detection_config.json,structure_config.json,inference.py,postprocess.py,model.py,inference_handler.py,inference.py" --export-path model_store
RUN torch-model-archiver --model-name table_transformer_structure --version 1.0 --model-file my_model/pubtables1m_structure_detr_r18.pth --handler inference_handler:table_transformer_model_handler --extra-files "detection_config.json,structure_config.json,inference.py,postprocess.py,model.py,inference_handler.py,inference.py" --export-path model_store

RUN echo "Archive created!"

# Production image
FROM pytorch/torchserve:latest-gpu

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD docker/entrypoint.sh entrypoint.sh

COPY --from=builder /usr/app/model_store model_store

ADD scripts/start-torchserve.sh start-torchserve.sh

COPY config.properties config.properties

CMD ["sh ./start-torchserve.sh"]

