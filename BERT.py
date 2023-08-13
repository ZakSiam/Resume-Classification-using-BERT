# Importing all the required packages or libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import set_seed
import torch
from torch.utils.data import Dataset

seed_val = 42

# Setting the seed value for reproducibility
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
set_seed(seed_val)

# Loading the Resume.csv dataset
file_path = './data/Resume.csv'
data = pd.read_csv(file_path)
data = data[['Resume_str', 'Category']]

# Examining the dataset provided and understanding the distribution of the different categories
category_distribution = data['Category'].value_counts()
print(category_distribution)
category_distribution.plot(kind='bar')
plt.title('Distribution of Categories')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()

# Splitting the dataset into training, validation, and test sets
X = data['Resume_str']
y = data['Category'].factorize()[0] # Conversion of categories to integers

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialization of tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Converting data to PyTorch Dataset using the ResumeDataset class
class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
        item = {key: inputs[key][0] for key in inputs} 
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Conversion of the Series objects to lists and creation of the custom dataset objects
train_dataset = ResumeDataset(X_train.tolist(), y_train.tolist())
val_dataset = ResumeDataset(X_val.tolist(), y_val.tolist())
test_dataset = ResumeDataset(X_test.tolist(), y_test.tolist())

# Initialization of the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(y)))

# Training arguments
training_args = TrainingArguments(
    output_dir='./BERT Results/results',
    num_train_epochs=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./BERT Results/logs',
    logging_steps=100,
    evaluation_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    report_to="tensorboard", # Enabling TensorBoard logging for additional insights or visualizations on the model's performance
    seed=seed_val,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluation on the test set
test_results = trainer.predict(test_dataset)

# Computation of different metrics on the test set
test_preds = test_results.predictions.argmax(axis=1)
accuracy = accuracy_score(y_test, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted')
print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}")

# Saving the best BERT model for further inference later
model.save_pretrained('./BERT Results/saved_model')
