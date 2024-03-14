import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Extended Label Encoder to handle unseen labels
class ExtendedLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, y):
        self.classes_, y = np.unique(y, return_inverse=True)
        self.classes_ = np.append(self.classes_, 'unknown')
        return self

    def transform(self, y):
        unseen_mask = ~np.isin(y, self.classes_)
        y_transformed = np.where(unseen_mask, 'unknown', y)
        _, y_transformed = np.unique(y_transformed, return_inverse=True)
        return y_transformed

    def inverse_transform(self, y):
        return self.classes_[y]

# Dataset
class GdprDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create data loader
def create_data_loader(X, y, tokenizer, max_len, batch_size):
    ds = GdprDataset(
        texts=X,
        labels=y,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=16)

# Model
class GdprClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(GdprClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)



# Evaluation
def eval_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            true_labels.extend(labels)

    predictions = torch.stack(predictions).cpu()
    true_labels = torch.stack(true_labels).cpu()
    return accuracy_score(true_labels, predictions)