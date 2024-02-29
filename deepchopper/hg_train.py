import numpy as np
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from .data.hg_data import load_and_split_dataset
from .models.hyena import (
    HyenaDNAForTokenClassification,
    load_config_and_tokenizer_from_hyena_model,
    tokenize_dataset,
)


def save_predicts(predicts, output_dir):
    np.save(output_dir / "predicts.npy", predicts)


def train():
    model_name = "hyenadna-small-32k-seqlen"
    data_file = {"train": "./tests/data/test_input.parquet"}
    learning_rate = 2e-5
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    num_train_epochs = 20
    weight_decay = 0.01
    torch_compile = False
    output_dir = "hyena_model_train"
    push_to_hub = False

    tokenizer, model_config = load_config_and_tokenizer_from_hyena_model(model_name)
    train_dataset, val_dataset, test_dataset = load_and_split_dataset(data_file)

    tokenize_train_dataset = tokenize_dataset(
        train_dataset, tokenizer, max_length=model_config.max_seq_len
    )
    tokenize_val_dataset = tokenize_dataset(
        val_dataset, tokenizer, max_length=model_config.max_seq_len
    )
    tokenize_test_dataset = tokenize_dataset(
        test_dataset, tokenizer, max_length=model_config.max_seq_len
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    model = HyenaDNAForTokenClassification(model_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=push_to_hub,
        torch_compile=torch_compile,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenize_train_dataset,
        eval_dataset=tokenize_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    trainer.predict(tokenize_test_dataset)


if __name__ == "__main__":
    train()
