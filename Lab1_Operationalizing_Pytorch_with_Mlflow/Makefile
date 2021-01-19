clean: ## Clean
	rm -rf logs
	rm -rf model_store
	rm -rf artifacts
	rm -rf labml_logs
	rm mlflow.db

train: export MLFLOW_TRACKING_URI:=http://localhost:5005
train: export PYTHONPATH=$$PYTHONPATH:./src
train:
	python -m bert_classifier.train

model_store: # Create empty model store
	mkdir model_store

torchserve-start: model_store ## Start torch serve
	torchserve --start --model-store model_store

torchserve-stop: ## Stop torch serve
	torchserve --stop

deploy: export MLFLOW_TRACKING_URI:=http://localhost:5005
deploy:  ## Deploy trained model make deploy MODEL='/BertModel/6'
	mlflow deployments create -t torchserve \
		-m "models:${MODEL}" \
		--name news_classification_test \
		-C "MODEL_FILE=src/bert_classifier/train.py" \
		-C "HANDLER=src/bert_classifier/handler.py"

mlflow-ui:	## Start MLFlow UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./artifacts

mlflow-server: ## Start MLFlow server
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./artifacts \
		--host localhost \
		--port 5005

predict: export PYTHONPATH=$$PYTHONPATH:./src
predict: ## Make a prediction
	PYTHONPATH=$$PYTHONPATH:./src python -m bert_classifier.predict

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: clean torchserve-start torchserve-stop deploy predict help env
.DEFAULT_GOAL := help
