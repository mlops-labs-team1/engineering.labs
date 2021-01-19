"""
Not used anymore.
Leaving for learning purposes.
"""
import mlflow
from mlflow.deployments import get_deploy_client


def deploy():
    # This can be set as an environment variable
    mlflow.set_tracking_uri("http://localhost:5005")
    client = get_deploy_client('torchserve')
    # path = Path('').absolute() / 'models'
    # client.create_deployment('news_classification_test', f'file://{path}',
    client.create_deployment('news_classification_test', f'models:/BertModel/2',
                             config={
                                 'MODEL_FILE': 'src/bert_classifier/train.py',
                                 'HANDLER': 'src/bert_classifier/handler.py'
                             })


if __name__ == '__main__':
    deploy()
