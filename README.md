
# Documentation

The data have been downloaded from the following link.
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

BERT.py describes the visualization of the Category classes, data preprocessing techniques, BERT model training and testing, and also reports different evaluation metrics on the test set.

categories_mapping.py factorizes the categories in the dataset and then saves the mapping for further use in script.py. 

The BERT (Bidirectional Encoder Representations from Transformers) model has been chosen for this task primarily because of its state-of-the-art performance, bidirectional understanding, pre-trained knowledge and fine-tuning capability.

The data have been divided into training, validation, and test sets where training, validation and test sets consist of 60%, 20% and 20% of the original dataset, respectively.

Different preprocessing or feature extraction methods have been employed. The command BertTokenizer.from_pretrained('bert-base-uncased') has been used to tokenize the resumes into tokens which can be processed by the BERT model. The tokenizer further manages padding (padding='max_length'), truncation (truncation=True), and conversion to tensors (return_tensors="pt"), thus making the text consistent and suitable for the BERT model. This process is executed inside the ResumeDataset class, particularly in the __getitem__ method.






## Deployment

Installation

```bash
  pip install -r requirements.txt
``` 

First unzip Data.zip and Script Development.zip files. 

Download the trained BERT model results ('BERT Results.zip' file) from this link.

https://drive.google.com/file/d/1UW8PD3caX7FwEUj487WnnolM2cS-KI7m/view?usp=drive_link

Instructions on how to run the script and expected outputs.

```bash
  python script.py "./Script Development/Test Input"
```

After extracting the Script Development.zip file, inside the Script Development folder, the Test Input directory contains a bunch of resumes in PDF format. Using the saved best BERT model (which can be found inside the directory './BERT Results/saved_model' after extracting the 'BERT Results.zip' file), the predicted results in both directory structures and csv format are stored in the './Script Development/Sample Output' directory.

For getting a sample output, the ‘Test Input’ directory was created that contains a total of 50 resumes in pdf format (as a sample test set), which were randomly collected from the original dataset given in pdf format.

The prediction accuracy of the best BERT model for these 50 resumes from the input directory, while compared to the corresponding ground truth categories, is approximately 80%.

For the visualization of the training progress, one can use the following command while training the BERT model using the 'BERT.py' file.

```bash
  tensorboard ---logdir "./BERT Results/logs"
```

