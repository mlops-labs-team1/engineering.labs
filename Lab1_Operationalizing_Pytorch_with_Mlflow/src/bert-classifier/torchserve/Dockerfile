FROM gcr.io/engineeringlab/base:latest

ARG GCP_CREDS_JSON_BASE64

ENV MLFLOW_HOME /opt/mlflow
ENV MODEL_STORE=${MLFLOW_HOME}/model_store
ENV MLFLOW_TRACKING_URI=http://35.185.111.8:5000

ENV GCP_CREDS_JSON_BASE64_ENV=${GCP_CREDS_JSON_BASE64}
ENV GOOGLE_APPLICATION_CREDENTIALS=${MLFLOW_HOME}/gcp.json

RUN mkdir -p ${MLFLOW_HOME}/scripts && \
    mkdir -p ${MODEL_STORE} && \
    echo "${GCP_CREDS_JSON_BASE64_ENV}"| base64 --decode > "${GOOGLE_APPLICATION_CREDENTIALS}" && \
    git clone https://github.com/pytorch/serve.git && \
    cd serve && python ./ts_scripts/install_dependencies.py && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY torchserve/entrypoint.sh ${MLFLOW_HOME}/scripts/run.sh
COPY torchserve/config.properties ${MLFLOW_HOME}/config.properties
RUN chmod +x ${MLFLOW_HOME}/scripts/run.sh

COPY torchserve/news_classifier_handler.py ${MLFLOW_HOME}/
# TODO: This should be taken from MLFlow
COPY trainer/news_classifier.py ${MLFLOW_HOME}/

VOLUME ["${MLFLOW_HOME}/scripts/"]

WORKDIR ${MLFLOW_HOME}

EXPOSE 7070
EXPOSE 7071
EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

ARG MODEL_NAME
RUN torchserve --start --ts-config ${MLFLOW_HOME}/config.properties && \
    mlflow deployments create \
        -t torchserve \
        -m models:/${MODEL_NAME} \
        --name news_classification \
        -C "MODEL_FILE=news_classifier.py" \
        -C "HANDLER=news_classifier_handler.py"\
        -C "EXPORT_PATH=${MODEL_STORE}" && \
    torchserve --stop

ENTRYPOINT ["./scripts/run.sh"]
CMD ["serve"]
