import argparse
import os
import shutil

import mlflow.pytorch
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
from labml import experiment, tracker, monit
from labml.logger import inspect
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets.text_classification import URLS
from torchtext.utils import download_from_url, extract_archive
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)

class_names = ["World", "Sports", "Business", "Sci/Tech"]
RANDOM_SEED = 42


class AGNewsDataset(Dataset):
    """
    Constructs the encoding with the dataset
    """

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Dropout(p=0.2)
        self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir='.cache')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.out = nn.Linear(512, len(class_names))

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Input sentences from the batch
        :param attention_mask: Attention mask returned by the encoder

        :return: output - label for the input text
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = F.relu(self.fc1(outputs.pooler_output))
        output = self.drop(output)
        output = self.out(output)
        return output


class NewsClassifierTrainer:
    def __init__(self, *, epochs: int, n_samples: int, vocab_file_url: str, is_save_model: bool, model_path: str,
                 batch_size: int = 16,
                 max_len: int = 160):
        self.model_path = model_path
        self.is_save_model = is_save_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_len = max_len
        self.n_samples = n_samples
        self.vocab_file_url = vocab_file_url
        self.vocab_file = "bert_base_uncased_vocab.txt"

        self.df = None
        self.tokenizer = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.optimizer = None
        self.total_steps = None
        self.scheduler = None
        self.loss_fn = None

    @staticmethod
    def process_label(rating):
        rating = int(rating)
        return rating - 1

    @staticmethod
    def create_data_loader(df, tokenizer, max_len, batch_size):
        """
        :param df: DataFrame input
        :param tokenizer: Bert tokenizer
        :param max_len: maximum length of the input sentence
        :param batch_size: Input batch size

        :return: output - Corresponding data loader for the given input
        """
        ds = AGNewsDataset(
            reviews=df.description.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return DataLoader(ds, batch_size=batch_size, num_workers=4)

    def prepare_data(self):
        """
        Creates train, valid and test data loaders from the csv data
        """
        dataset_tar = download_from_url(URLS["AG_NEWS"], root=".data")
        extracted_files = extract_archive(dataset_tar)

        train_csv_path = None
        for file_name in extracted_files:
            if file_name.endswith("train.csv"):
                train_csv_path = file_name

        self.df = pd.read_csv(train_csv_path)

        self.df.columns = ["label", "title", "description"]
        self.df.sample(frac=1)
        self.df = self.df.iloc[: self.n_samples]

        self.df["label"] = self.df.label.apply(self.process_label)

        if not os.path.isfile(self.vocab_file):
            file_pointer = requests.get(self.vocab_file_url, allow_redirects=True)
            if file_pointer.ok:
                with open(self.vocab_file, "wb") as f:
                    f.write(file_pointer.content)
            else:
                raise RuntimeError("Error in fetching the vocab file")

        self.tokenizer = BertTokenizer(self.vocab_file)

        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        self.df_train, self.df_test = train_test_split(
            self.df, test_size=0.1, random_state=RANDOM_SEED, stratify=self.df["label"]
        )
        self.df_val, self.df_test = train_test_split(
            self.df_test, test_size=0.5, random_state=RANDOM_SEED, stratify=self.df_test["label"]
        )

        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.max_len, self.batch_size
        )
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.max_len, self.batch_size
        )
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.max_len, self.batch_size
        )

    def set_optimizer(self, model):
        """
        Sets the optimizer and scheduler functions
        """
        self.optimizer = AdamW(model.parameters(), lr=1e-3, correct_bias=False)
        self.total_steps = len(self.train_data_loader) * self.epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def save_model(self, model):
        with monit.section('Save model'):
            mlflow.pytorch.log_model(model, "bert-model",
                                     registered_model_name="BertModel",
                                     extra_files=["class_mapping.json", "bert_base_uncased_vocab.txt"])
            # if os.path.exists(self.model_path):
            #     shutil.rmtree(self.model_path)
            # mlflow.pytorch.save_model(
            #     model,
            #     path=self.model_path,
            #     requirements_file="requirements.txt",
            #     extra_files=["class_mapping.json", "bert_base_uncased_vocab.txt"],
            # )

    def start_training(self, model):
        """
        Initializes the Training step with the model initialized

        :param model: Instance of the NewsClassifier class
        """
        best_loss = float('inf')

        for epoch in monit.loop(self.epochs):
            with tracker.namespace('train'):
                self.train_epoch(model, self.train_data_loader, 'train')

            with tracker.namespace('valid'):
                _, val_loss = self.train_epoch(model, self.val_data_loader, 'valid')

            if val_loss < best_loss:
                best_loss = val_loss

                if self.is_save_model:
                    self.save_model(model)

            tracker.new_line()

    def train_epoch(self, model: nn.Module, data_loader: DataLoader, name: str):
        """
        Train/Validate for an epoch
        """

        model.train(name == 'train')
        correct_predictions = 0
        total = 0
        total_loss = 0

        with torch.set_grad_enabled(name == 'train'):
            for i, data in monit.enum(name, data_loader):
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item() * len(preds)

                correct_predictions += torch.sum(preds == targets).item()
                total += len(preds)
                tracker.add('loss.', loss)
                if name == 'train':
                    tracker.add_global_step(len(preds))

                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (i + 1) % 10 == 0:
                    tracker.save()

        tracker.save('accuracy.', correct_predictions / total)
        mlflow.log_metric(f"{name}_acc", float(correct_predictions / total), step=tracker.get_global_step())
        mlflow.log_metric(f"{name}_loss", float(total_loss / total), step=tracker.get_global_step())

        return correct_predictions / total, total_loss / total

    def get_predictions(self, model, data_loader):
        """
        Prediction after the training step is over

        :param model: Instance of the NewsClassifier class
        :param data_loader: Data loader for either test / validation dataset

        :result: output - Returns prediction results,
                          prediction probablities and corresponding values
        """
        model = model.eval()

        review_texts = []
        predictions = []
        prediction_probs = []
        real_values = []

        with torch.no_grad():
            for d in data_loader:
                texts = d["review_text"]
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                probs = F.softmax(outputs, dim=1)

                review_texts.extend(texts)
                predictions.extend(preds)
                prediction_probs.extend(probs)
                real_values.extend(targets)

        predictions = torch.stack(predictions).cpu()
        prediction_probs = torch.stack(prediction_probs).cpu()
        real_values = torch.stack(real_values).cpu()
        return review_texts, predictions, prediction_probs, real_values


def main():
    parser = argparse.ArgumentParser(description="PyTorch BERT Example")

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="N",
        help="batch size (default: 16)",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=160,
        metavar="N",
        help="number of tokens per sample (rest is truncated) (default: 140)",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1_000,
        metavar="N",
        help="Number of samples to be used for training "
             "and evaluation steps (default: 15000) Maximum:100000",
    )

    parser.add_argument(
        "--save_model",
        type=bool,
        default=True,
        help="For Saving the current Model",
    )

    parser.add_argument(
        "--vocab_file",
        default="https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        help="Custom vocab file",
    )

    parser.add_argument(
        "--model_save_path", type=str, default="models", help="Path to save mlflow model"
    )

    experiment.create(name='bert_news')
    args = parser.parse_args()
    experiment.configs(args.__dict__)

    mlflow.set_tracking_uri("http://localhost:5005")
    mlflow.start_run()
    mlflow.log_param("epochs", args.max_epochs)
    mlflow.log_param("samples", args.num_samples)

    with experiment.start():
        trainer = NewsClassifierTrainer(epochs=args.max_epochs,
                                        n_samples=args.num_samples,
                                        vocab_file_url=args.vocab_file,
                                        is_save_model=args.save_model,
                                        model_path=args.model_save_path,
                                        batch_size=args.batch_size,
                                        max_len=args.max_len)
        model = Model()
        model = model.to(trainer.device)
        trainer.prepare_data()
        trainer.set_optimizer(model)
        trainer.start_training(model)

        with tracker.namespace('test'):
            test_acc, test_loss = trainer.train_epoch(model, trainer.test_data_loader, 'test')

        y_review_texts, y_pred, y_pred_probs, y_test = trainer.get_predictions(
            model, trainer.test_data_loader
        )

        inspect(y_review_texts)
        inspect(torch.stack((y_pred, y_test), dim=1))

        mlflow.log_metric("test_acc", float(test_acc), step=tracker.get_global_step())
        mlflow.log_metric("test_loss", float(test_loss), step=tracker.get_global_step())

        mlflow.end_run()


if __name__ == '__main__':
    main()
