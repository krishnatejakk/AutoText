from copy import copy, deepcopy
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, get_scheduler, BertConfig, AdamW
from bert_mlp import CustomBERTModel
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from selectionstrategies.supervisedlearning import OMPGradMatchStrategy


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
model1 = CustomBERTModel(configuration)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model1.to(device)
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

criterion = CrossEntropyLoss(reduction='mean')
criterion_nored = CrossEntropyLoss(reduction='none')

fraction = 0.3
budget = int(fraction * len(train_dataset))
setf_model = OMPGradMatchStrategy(train_dataloader, val_dataloader, model1, criterion_nored,
                                  0.01, device, configuration.num_classes, True, 'PerBatch',
                                  False, True, lam=0, eps=1e-100)
model_state_dict = deepcopy(model.state_dict())
idxs, gammas = setf_model.select(budget, model_state_dict)
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, finetune=False)      
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

