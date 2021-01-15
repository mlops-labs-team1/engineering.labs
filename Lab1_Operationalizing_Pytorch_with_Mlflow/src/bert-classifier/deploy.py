from pathlib import Path

import mlflow
from mlflow.deployments import get_deploy_client


def deploy():
    mlflow.set_tracking_uri("http://localhost:5005")
    client = get_deploy_client('torchserve')
    path = Path('').absolute() / 'models'
    # client.create_deployment('news_classification_test', f'file://{path}',
    client.create_deployment('news_classification_test', f'models:/BertModel/6',
                             config={
                                 'MODEL_FILE': 'news_classifier.py',
                                 'HANDLER': 'news_classifier_handler.py'
                             })


if __name__ == '__main__':
    deploy()
