import random

from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_metric
import numpy as np
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTConfig,
    TrainingArguments,
    Trainer,
)
import torch
from torchvision import transforms
from tqdm import tqdm

DATASET_NAME = "Aerobotics/citrico_2615"
BASE_MODEL_NAME = "google/vit-base-patch16-224-in21k"
MODEL_OUTPUT_DIR = "./citrico_2615_vit/vit-base-citrco-2615"


def load_processor() -> ViTImageProcessor:
    processor = ViTImageProcessor.from_pretrained(BASE_MODEL_NAME)
    return processor


# Globals
processor = load_processor()
metric = load_metric("accuracy")


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


def train_model():
    print("Starting preparation for training")
    ds = load_hugging_face_dataset()
    print("Augmenting dataset")
    ds_augmented = ds.map(data_augmentations)
    prepared_ds = load_train_prepped_dataset(ds_augmented)
    print("Finished preparation for training")

    # Model prep
    labels = ds["train"].features["axis_label"].names
    config = ViTConfig.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
    )
    model = ViTForImageClassification.from_pretrained(BASE_MODEL_NAME, config=config)

    # Training and eval
    trainer = init_trainer(model, prepared_ds)
    run_training(trainer)
    run_eval(trainer, prepared_ds)
    print("Completed model training")


def data_augmentations(example):
    augmentation_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.GaussianBlur(kernel_size=3),
            # Consider other augmentations
            transforms.ToTensor(),
            transforms.normalize,
            transforms.ColorJitter(),
        ]
    )
    example["image"] = augmentation_transforms(example["image"])
    return example


def load_hugging_face_dataset() -> Dataset:
    ds = load_dataset(DATASET_NAME)
    return ds


def load_train_prepped_dataset(ds: Dataset) -> DatasetDict:
    prepared_ds = ds.with_transform(transform)
    return prepared_ds


def process_example(example):
    inputs = processor(example["image"], return_tensors="pt")
    inputs["labels"] = example["axis_label"]
    return inputs


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["axis_label"]
    return inputs


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def init_trainer(
    model: ViTForImageClassification,
    prepared_ds: DatasetDict,
) -> Trainer:
    # Update training params as needed
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=5,
        fp16=False,
        save_steps=50,
        eval_steps=50,
        logging_steps=5,
        learning_rate=1.75e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["validation"],
        tokenizer=processor,
    )


def run_training(trainer: Trainer):
    print("Starting training")
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print("Finished training")


def run_eval(trainer: Trainer, prepared_ds: DatasetDict):
    print("Starting evaluation")
    metrics = trainer.evaluate(prepared_ds["validation"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print("Finished evaluation")


if __name__ == "__main__":
    train_model()
