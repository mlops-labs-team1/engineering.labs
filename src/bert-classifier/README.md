# Deploying BERT - News Classification using TorchServe

The code, adapted from this [repository](https://github.com/mlflow/mlflow-torchserve/blob/master/examples/BertNewsClassification),
is dedicated to model training (fine tuning) as in the original example, but we have included the necessary files to deploy the model to a torchserve docker image.  
It would be worthwhile working through that repository first to understand the concepts before going through ours.  
The Pytorch model including the extra files such as the vocabulary file and class mapping file, which are essential to make the model functional,
are saved to a MLflow tracking server using the function `mlflow.pytorch.log_model`. This requires a tracking server to be set up with a sql backend before starting.

By default,  the script exports the model file as `model.pth`.
This example workflow includes the following steps,
1. A pre trained Hugging Face bert model is fine-tuned to classify news.
2. Model is saved with extra files model,summary, parameters and extra files at the end of training to MLflow server
3. Deployment of the  model in a Torchserve Docker image.

Since training and serving are equally important to us in this project, we've split the directory into the files needed 
for model training in the `trainer` directory and those needed for model serving in the `torchserve` directory.  
In order to keep training and serving environments as consistent as possible, we have a common `requirements.txt`
and `Dockerfile.base` for both parts.

## MLflow server
To allow the code to log model artifacts remotely, a tracking server needs to be set up with a 
sql [backend store](https://www.mlflow.org/docs/latest/tracking.html#backend-stores). 
Not required but advisable is setting up a bucket based [artifact store](https://www.mlflow.org/docs/latest/tracking.html#artifact-stores)

To configure MLflow to log to a custom (non-default) tracking location, set the MLFLOW_TRACKING_URI environment variable, e.g. via export MLFLOW_TRACKING_URI=http://localhost:5000/.
For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).

Our MLflow server used [Google Cloud Storage](https://www.mlflow.org/docs/latest/tracking.html#google-cloud-storage) as its artifact store
and so requires a [json credentials file](https://cloud.google.com/docs/authentication/getting-started) for a 
service account with access to the bucket. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the location of this file.  
This is because although we track metrics and metadata about the
model in the tracking server backend store, the artifacts are sent directly to the cloud bucket.

See the `infra` directory for more detailed info on how we set up our MLflow tracking server (and all our infrastructure)

## Package Requirement

Install MLflow per the [instructions](https://github.com/mlflow/mlflow#installing).  
This project doesn't require `conda` on the `PATH`, but does require `docker` to be installed.

## Training
### Building the training environment
To run the code reproducibly, we use the `docker` project environment.
See the [documentation](https://www.mlflow.org/docs/latest/projects.html#project-environments).  

#### Building the base image
A lot of the bugs that arose during this project came from the training and serving containers having different python environments.
To help prevent this we created a base image with the required python environment from which the training and serving images will start from.

To build the image from the current directory run
```
docker build -t gcr.io/engineeringlab/base:latest -f Dockerfile.base .
```
#### Building the training image
The image contains the required GCP credentials for logging models. This is passed as a build argument using base64 encoding.

To build the image from the current directory run
```
docker build --build-arg GCP_CREDS_JSON_BASE64="$(base64 $GOOGLE_APPLICATION_CREDENTIALS)" -f ./trainer/Dockerfile .
```

### Running the code
To run the example via MLflow, navigate to the `src/bert-classifier/trainer` directory.

Edit the `MLProject` file and replace `##IMAGE##` with `latest`. (side note: this is replaced in the github actions workflow to create a unique training image during CI)

Run the command
```
mlflow run .
```

This will run `news_classifier.py` with the default set of parameters defined in the `MLProject` file 
such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

#### Passing custom training parameters

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train models. Training can be interrupted early via Ctrl+C
2. num_samples -Number of input samples required for training

For example:
```
mlflow run . -P max_epochs=5 -P num_samples=50000
```
### Registering the model
To register a model in the [MLflow Model Registry](https://www.mlflow.org/docs/latest/model-registry.html#) from your experiment,
pass the `model_name` parameter.  
```
mlflow run . -P model_name=BertModel
```
This will create a new registered model with that name if one doesn't exist, or will create a new version of 
the registered model if it does exist.

## Serving
### Building the serving image
To serve the code reproducibly (and particularly to ensure the training and serving environments match, which is particularly important as the model files are pickled)
we build a docker image for serving.
To build an image for a registered model, e.g. `BertModel` version `7`, from this directory run
```
docker build --build-arg GCP_CREDS_JSON_BASE64="$(base64 $GOOGLE_APPLICATION_CREDENTIALS)" --build-arg MODEL_NAME=BertModel/7 -f ./torchserve/Dockerfile .
```
These images can be tagged with the model name and version and pushed to a registry for testing or deployment.
```
docker build -t gcr.io/engineeringlab/torchserve:BertModel-7 --build-arg GCP_CREDS_JSON_BASE64="$(base64 $GOOGLE_APPLICATION_CREDENTIALS)" --build-arg MODEL_NAME=BertModel/7 -f ./torchserve/Dockerfile .
docker push gcr.io/engineeringlab/torchserve:BertModel-7
```

To build the image containing the model, we start the torchserve server, use the new [mlflow-torchserve](https://github.com/mlflow/mlflow-torchserve) plugin to 
add the model to the torchserve server, then stop the server.
#### Starting TorchServe

The following command is run in the Dockerfile to start torchserve. This reads the `config.properties` file to set the `model_store` location.

`torchserve --start --ts-config ${MLFLOW_HOME}/config.properties`

The same command is used as the entrypoint for the container, which loads up all models stored in the `model_store` directory, 
and binds the inference port to the external world.

#### Creating a new deployment

Once torchserve has started, the following command is run in the Dockerfile to create a new deployment named `news_classification` from a registered model, e.g. `BertModel` version `7`

```mlflow deployments create -t torchserve -m models:/BertModel/7 --name news_classification -C "MODEL_FILE=news_classifier.py" -C "HANDLER=news_classifier_handler.py" -C "EXPORT_PATH=/opt/mlflow/model_store"```

This downloads the model artifacts from MLflow, create a torchserve model archive `.mar` file, stores this in
the `model_store` directory, and registers the model with torchserve.

Torchserve deployment plugin has the ability to detect and add the `requirements.txt` and the extra files. And hence, during the
mar file generation, TorchServe automatically bundles the `requirements.txt`and extra files along with the model.  
Torchserve doesn't use the `requirements.txt` file by default, we would need to have set `install_py_dep_per_model=true` in `config.properties`.
Since we are running from an image with the requirements installed and only run one model from each container, 
we don't need to take advantage of this aspect of the deployment plugin.
