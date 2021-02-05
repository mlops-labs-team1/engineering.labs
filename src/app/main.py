from flask import Flask, request, render_template
import requests

app = Flask(__name__)
app.config.from_object("settings")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    model_serve_url = app.config["SERVE_URI"]
    if model_serve_url is None:
        return render_template('index.html', prediction_text="ERROR: Model serve url not set")

    news_text = request.form["news"]

    try:
        resp = requests.post(f"{model_serve_url}/predictions/news_classification", data=news_text)
    except Exception as e:
        return render_template('index.html', prediction_text=str(e), input=news_text)

    if resp.ok:
        output = f"The article is of type {resp.text}"
    else:
        output = f"Error in model response:\n{resp.text}"

    return render_template('index.html', prediction_text=output, input=news_text)


if __name__ == "__main__":
    app.run()
