clean: ## Clean
	rm -rf models
	rm -rf logs
	rm -rf mlruns
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
	mlflow deployments create -t torchserve \
		-m 'file://$(shell pwd)/models' \
		--name news_classification_test \
		-C "MODEL_FILE=news_classifier.py" \
		-C "HANDLER=news_classifier_handler.py"

predict: ## Make a prediction
	mlflow deployments predict \
		--name news_classification_test \
		--target torchserve \
		--input-path input.json \
		--output-path output.json
	cat output.json

help: ## Show this help.
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'

.PHONY: clean torchserve-start torchserve-stop deploy predict help
.DEFAULT_GOAL := help
