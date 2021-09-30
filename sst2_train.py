from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, get_scheduler, BertConfig, AdamW
from bert_mlp import CustomBERTModel
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def tokenize_function(example):
    return tokenizer(example["sentence"], padding = 'max_length', truncation=True)


def compute_metrics(eval_preds):
    metric = load_metric("accuracy", "f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


configuration = BertConfig()
setattr(configuration, 'l1', 512)
setattr(configuration, 'num_classes', 2)


raw_datasets = load_dataset("glue", "sst2")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
val_dataset = tokenized_datasets["validation"].shuffle(seed=42)
tst_dataset = tokenized_datasets["test"].shuffle(seed=42)

train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)
tst_dataloader = DataLoader(tst_dataset, batch_size=8)

model = CustomBERTModel(configuration)
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

criterion = CrossEntropyLoss(reduction='mean')
                
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, finetune=True)      
        loss = criterion(outputs, batch['labels'].view(-1))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric= load_metric("accuracy")
model.eval()
for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    # logits = outputs.logits
    predictions = torch.argmax(outputs, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
print(metric.compute())

