#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

clone() {
    TMPDIR=$(mktemp -d /tmp/model.XXXXXXXXXX)
    cd "$TMPDIR"
    git clone https://github.com/mlops-labs-team1/engineering.labs.git
    cd engineering.labs/src/bert-classifier/trainer/
}

export_model_version() {
    JSON_FILE="$1"

    NAME=$(jq -r '.name' <"$JSON_FILE")
    VERSION=$(jq -r '.version' <"$JSON_FILE")
    MODEL_NAME="${NAME}/$VERSION"
    SERVE_IMAGE_TAG=$(echo "$MODEL_NAME" | tr / -)

    echo '::set-output name=model_name::'"$MODEL_NAME"
    echo '::set-output name=serve_image_tag::'"$SERVE_IMAGE_TAG"
}

train() {
    TAG="$1"

    JSON_FILE="$TAG.json"

    sed -i -e s/##IMAGE##/"$TAG"/ MLproject
    mlflow run --no-conda -P json_dump="/tmp/englab/$JSON_FILE" -P model_name=BertModel .

    export_model_version "$JSON_FILE"
}

main() {
    export MLFLOW_TRACKING_URI="$1"
    SHA="$2"

    clone
    train "$SHA"
}

main "$@"
