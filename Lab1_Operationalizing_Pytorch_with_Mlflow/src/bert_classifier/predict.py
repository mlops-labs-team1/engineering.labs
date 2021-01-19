import json

from labml.logger import inspect
from mlflow.deployments import get_deploy_client


def predict():
    data = ["This year business is good", "Fortnite, Football And Soccer, And Their Surprising Similarities"]
    client = get_deploy_client('torchserve')
    for d in data:
        data_json = json.dumps({'data': [d], 'uuid': 'str'})
        res = client.predict('news_classification_test', data_json)
        inspect(text=d, category=res)


if __name__ == '__main__':
    predict()
