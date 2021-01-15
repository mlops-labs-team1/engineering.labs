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
    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.EPOCHS = args.max_epochs
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
        self.BATCH_SIZE = 16
        self.MAX_LEN = 160
        self.NUM_SAMPLES_COUNT = args.num_samples
        self.VOCAB_FILE_URL = args.vocab_file
        self.VOCAB_FILE = "bert_base_uncased_vocab.txt"

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
        Creates train, valid and test dataloaders from the csv data
        """
        dataset_tar = download_from_url(URLS["AG_NEWS"], root=".data")
        extracted_files = extract_archive(dataset_tar)

        train_csv_path = None
        for fname in extracted_files:
            if fname.endswith("train.csv"):
                train_csv_path = fname

        self.df = pd.read_csv(train_csv_path)

        self.df.columns = ["label", "title", "description"]
        self.df.sample(frac=1)
        self.df = self.df.iloc[: self.NUM_SAMPLES_COUNT]

        self.df["label"] = self.df.label.apply(self.process_label)

        if not os.path.isfile(self.VOCAB_FILE):
            filePointer = requests.get(self.VOCAB_FILE_URL, allow_redirects=True)
            if filePointer.ok:
                with open(self.VOCAB_FILE, "wb") as f:
                    f.write(filePointer.content)
            else:
                raise RuntimeError("Error in fetching the vocab file")

        self.tokenizer = BertTokenizer(self.VOCAB_FILE)

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        self.df_train, self.df_test = train_test_split(
            self.df, test_size=0.1, random_state=RANDOM_SEED, stratify=self.df["label"]
        )
        self.df_val, self.df_test = train_test_split(
            self.df_test, test_size=0.5, random_state=RANDOM_SEED, stratify=self.df_test["label"]
        )

        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.BATCH_SIZE
        )

    def set_optimizer(self, model):
        """
        Sets the optimizer and scheduler functions
        """
        self.optimizer = AdamW(model.parameters(), lr=1e-3, correct_bias=False)
        self.total_steps = len(self.train_data_loader) * self.EPOCHS

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps
        )

        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def start_training(self, model, args):
        """
        Initialzes the Traning step with the model initialized

        :param model: Instance of the NewsClassifier class
        """
        best_accuracy = 0

        for epoch in monit.loop(self.EPOCHS):
            self.train_epoch(model)

            with tracker.namespace('valid'):
                val_acc = self.eval_model(model, self.val_data_loader)

            if val_acc > best_accuracy:
                if args.save_model:
                    with monit.section('Save model'):
                        if os.path.exists(args.model_save_path):
                            shutil.rmtree(args.model_save_path)
                        mlflow.pytorch.save_model(
                            model,
                            path=args.model_save_path,
                            requirements_file="requirements.txt",
                            extra_files=["class_mapping.json", "bert_base_uncased_vocab.txt"],
                        )
                best_accuracy = val_acc

            tracker.new_line()

    def train_epoch(self, model):
        """
        Training process happens and accuracy is returned as output

        :param model: Instance of the NewsClassifier class

        :result: output - Accuracy of the model after training
        """

        model.train()
        correct_predictions = 0
        total = 0

        for i, data in monit.enum('Train', self.train_data_loader):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets).item()
            total += len(preds)
            tracker.add('loss.train', loss)
            tracker.add_global_step(len(preds))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            if (i + 1) % 10 == 0:
                tracker.save()

        tracker.save('accuracy.train', correct_predictions / total)

    def eval_model(self, model, data_loader):
        """
        Validation process happens and validation / test accuracy is returned as output

        :param model: Instance of the NewsClassifier class
        :param data_loader: Data loader for either test / validation dataset

        :result: output - Accuracy of the model after testing
        """
        model.eval()

        correct_predictions = 0
        total = 0

        with torch.no_grad():
            for d in monit.iterate('Valid', data_loader):
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                targets = d["targets"].to(self.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)

                loss = self.loss_fn(outputs, targets)
                correct_predictions += torch.sum(preds == targets).item()
                total += len(preds)
                tracker.add('loss.', loss)

        tracker.save('accuracy.', correct_predictions / total)
        return correct_predictions / total

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
        "--num_samples",
        type=int,
        default=120_000,
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

    with experiment.start():
        mlflow.start_run()

        trainer = NewsClassifierTrainer(args)
        model = Model()
        model = model.to(trainer.device)
        trainer.prepare_data()
        trainer.set_optimizer(model)
        trainer.start_training(model, args)

        print("TRAINING COMPLETED!!!")

        with tracker.namespace('test'):
            test_acc = trainer.eval_model(model, trainer.test_data_loader)
            print(test_acc)

        y_review_texts, y_pred, y_pred_probs, y_test = trainer.get_predictions(
            model, trainer.test_data_loader
        )

        mlflow.end_run()


if __name__ == '__main__':
    main()
