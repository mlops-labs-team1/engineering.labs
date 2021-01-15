clean: ## Clean
	rm -rf logs
	rm -rf model_store

train:
	python news_classifier.py

model_store: ## Create empty model store
	mkdir model_store

torchserve-start: model_store ## Start torch serve
	torchserve --start --model-store model_store

torchserve-stop: ## Stop torch serve
	torchserve --stop

deploy:  ## Deploy trained model
	python -m deploy

mlflow-ui:	## Start MLFlow UI
	mlflow ui --backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./artifacts

mlflow-server: ## Start MLFlow server
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./artifacts \
		--host localhost \
		--port 5005

predict: ## Make a prediction
	python -m predict

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: clean torchserve-start torchserve-stop deploy predict help
.DEFAULT_GOAL := help
