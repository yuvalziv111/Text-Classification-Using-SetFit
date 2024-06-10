import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import pyarrow as pa
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup, RobertaForSequenceClassification, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.metrics import precision_recall_fscore_support as score
from transformers import EarlyStoppingCallback
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import pipeline
from transformers.trainer_utils import number_of_arguments
from typing import Optional, Dict, Any, Callable
import sentencepiece
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.stem import PorterStemmer
from datasets import Dataset
import json
import pandas as pd
import numpy as np
import torch
import evaluate
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
import setfit
from setfit import SetFitModel, SetFitTrainer
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset


d = {}
class MyTrainer(Trainer):
    def __init__(self, **kwargs):
        super(MyTrainer, self).__init__(**kwargs)

    def evaluate(self, dataset: Optional[Dataset] = None, metric_key_prefix: str = "test") -> Dict[str, float]:
        """
        Computes the metrics for a given classifier.

        Args:
            dataset (`Dataset`, *optional*):
                The dataset to compute the metrics on. If not provided, will use the evaluation dataset passed via
                the `eval_dataset` argument at `Trainer` initialization.

        Returns:
            `Dict[str, float]`: The evaluation metrics.
        """

        if dataset is not None:
            self._validate_column_mapping(dataset)
            if self.column_mapping is not None:
                eval_dataset = self._apply_column_mapping(dataset, self.column_mapping)
            else:
                eval_dataset = dataset
        else:
            eval_dataset = self.eval_dataset

        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided to `Trainer.evaluate` nor the `Trainer` initialzation.")

        x_test = eval_dataset["text"]
        y_test = eval_dataset["label"]

        print("***** Running evaluation *****")
        y_pred = self.model.predict(x_test, use_labels=False)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu()

        # Normalize string outputs
        if y_test and isinstance(y_test[0], str):
            encoder = LabelEncoder()
            encoder.fit(list(y_test) + list(y_pred))
            y_test = encoder.transform(y_test)
            y_pred = encoder.transform(y_pred)


        final_results = []
        for metric in self.metric:
            metric_config = "multilabel" if self.model.multi_target_strategy is not None else None
            metric_fn = evaluate.load(metric, config_name=metric_config)
            if metric != 'accuracy':
                metric_kwargs = self.metric_kwargs
                results = metric_fn.compute(predictions=y_pred, references=y_test, **metric_kwargs)
            else:
                metric_kwargs = {}
                results = metric_fn.compute(predictions=y_pred, references=y_test, **metric_kwargs)
            final_results.append(results)
        if not isinstance(results, dict):
            results = {"metric": results}
        self.model.model_card_data.post_training_eval_results(
            {f"{metric_key_prefix}_{key}": value for key, value in results.items()}
        )
        return final_results

def plotting_results(train_loss,train_acc,val_loss,val_acc,epoch_num):
  fig, axs = plt.subplots(2, 2, figsize=(10,10))
  axs[0, 0].plot(epoch_num, train_loss)
  axs[0, 0].set_title("Train Loss")
  axs[1, 0].plot(epoch_num, train_acc)
  axs[1, 0].set_title("Train Accuracy")
  axs[0, 1].plot(epoch_num, val_loss,color="r")
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(epoch_num, val_acc,color="r")
  axs[1, 1].set_title("Test Accuracy")
  return fig.tight_layout()

def calculate_accuracy_per_topic(predicted_labels, true_labels, num_topics):
    topic_correct_counts = [0] * num_topics
    topic_sample_counts = [0] * num_topics

    for pred_label, true_label in zip(predicted_labels, true_labels):
        topic_sample_counts[true_label] += 1
        if pred_label == true_label:
            topic_correct_counts[true_label] += 1

    topic_accuracies = [correct_count / sample_count if sample_count > 0 else 0.0 for correct_count, sample_count in zip(topic_correct_counts, topic_sample_counts)]
    return topic_accuracies

class Create_Data(Dataset):
    def __init__(self, data_proc):
        self.data = data_proc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']  # Adjust column names if needed
        return line, label

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.column_names = ['text','label']

    def __getitem__(self, index):
        print('index: ',index)
        row = self.dataframe.iloc[index].to_numpy()
        features = row[1:]
        label = row[0]
        return features, label

    def __len__(self, dataframe):
        return len(self.dataframe)
    def rename_columns(self, new_column_names):
        self.column_names = new_column_names
        renamed_df = self.dataframe.rename(columns={old_name: new_name for old_name, new_name in zip(self.dataframe.columns, new_column_names)})
        self.dataframe = renamed_df

def custom_collate(batch):
  """Collate function that handles batches of different sizes."""
  batched_bigrams = []
  for bigram in batch:
    batched_bigrams.append(bigram[:len(bigram[0])])
  return batched_bigrams

# Defining the Class for tokenization and processing the text for input to the data_loader
class NLPModel(nn.Module):
    def __init__(self, tokenizer, max_seq_len=None, model=None):
        super(NLPModel, self).__init__()
        self.config = SetFitModel.from_pretrained(pretrained_model_name_or_path='andrr/setfit_retail', num_labels = len(df['label'].unique()), ignore_mismatched_sizes=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.model = model
        return

    def unfreeze_layers(self, num_layer):
        for ind, x in enumerate(self.modules()):
            # Comparing to feature extractor, now we are gonna unfreeze part of the layers and train our model using them
            if ind > num_layer:
                for param in x.parameters():
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask,labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
        loss = outputs.loss
        logits = outputs.logits
        prob = F.softmax(logits,dim=1)
        return prob,loss


        # Training function

def activate_train(data_loader, optimizer, scheduler, device):
    global ROBERTA_Class
    ROBERTA_Class.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0

    ### training regular
    for texts, labels in tqdm(data_loader, total=len(data_loader)):
        inputs = ROBERTA_Class.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=ROBERTA_Class.max_seq_len)
        true_labels += labels.numpy().flatten().tolist()
        optimizer.zero_grad()
        batch = {}

        batch = {'input_ids': inputs['input_ids'],
                              'attention_mask': inputs['attention_mask'],
                              'labels': torch.tensor(labels.type(torch.LongTensor))}
        outputs = ROBERTA_Class(**batch)
        loss = outputs[1]
        # logits = outputs[1:2][0][1]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ROBERTA_Class.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += outputs[0].argmax(axis=-1).flatten().tolist()



    accuracies = calculate_accuracy_per_topic(predictions_labels,true_labels,len(df['label'].unique()))
    avg_epoch_loss = total_loss / len(data_loader)
    return predictions_labels, true_labels, avg_epoch_loss,accuracies

# Function for validation
def validate(data_loader):
    global model
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    for epoch in epoch_list:
        for texts,labels in tqdm(data_loader):
            inputs = ROBERTA_Class.tokenizer(text=texts,
                                             return_tensors='pt',
                                             padding=True,
                                             truncation=True,
                                             max_length=ROBERTA_Class.max_seq_len)
            true_labels += labels.numpy().flatten().tolist()
            batch = {'input_ids': inputs['input_ids'],
                     'attention_mask': inputs['attention_mask'],
                     'labels': torch.tensor(labels.type(torch.LongTensor))}
            with torch.no_grad():
                outputs = ROBERTA_Class(**batch)
                loss = outputs[1]
                total_loss += loss.item()
                predictions_labels += outputs[0].argmax(axis=-1).flatten().tolist()
        avg_epoch_loss = total_loss / len(data_loader)
        print (fr"epoch: {epoch}, loss: {avg_epoch_loss}")
        accuracies = calculate_accuracy_per_topic(predictions_labels, true_labels, df['label'].unique())
        predicted = []
        true = []

        return predictions_labels, true_labels, avg_epoch_loss,accuracies
max_len = 350 # Max lenght of the text for input
batch_size = 8
epochs = 10
epoch_list = range(1, epochs+1)
device = torch.device("cuda")

# Load the data
df = pd.read_csv(fr'C:\Users\yuval\OneDrive\שולחן העבודה\text classification\Updated_train_set 07.11.23.csv', encoding='latin1')

df = df[df['topic'].isin(['on promotion', 'promotion ended'])]
def transform_by_mapping(x,mapping):
    label = mapping.loc[mapping['topic']==x]['label']
    return label

nltk.download('stopwords')
nltk.download('punkt')
def preprocess_text(text):
    # Tokenize the text
    if not isinstance(text, str):
        tokens = ""
    else:
        tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    remove_list = ['not','no','off','nor',"wasn't","haven't","now","isn't","hasn't","don't","doesn't","didn't","couldn't","aren't",'out','more','than']
    for item in remove_list:
        stop_words.remove(item)
    tokens = [token for token in tokens if token not in stop_words]

    # Remove punctuation marks
    table = str.maketrans('', '', string.punctuation)
    tokens =[token.translate(table) for token in tokens if token.translate(table)]

    ### Stemming
    stemmed = []
    stemmer = PorterStemmer()
    for token in tokens:
        if token.isalpha():

            word = stemmer.stem(token)
            stemmed.append(word)

    # Join the tokens back into a single string
    preprocessed_text = ' '.join(stemmed)

    return preprocessed_text

def calculate_accuracy_per_topic(predicted_labels, true_labels, num_topics):
    topic_correct_counts = [0] * num_topics
    topic_sample_counts = [0] * num_topics

    for pred_label, true_label in zip(predicted_labels, true_labels):
        topic_sample_counts[true_label] += 1
        if pred_label == true_label:
            topic_correct_counts[true_label] += 1

    topic_accuracies = [correct_count / sample_count if sample_count > 0 else 0.0 for correct_count, sample_count in zip(topic_correct_counts, topic_sample_counts)]
    topic_precisions = [correct_count / sample_count if sample_count > 0 else 0.0 for correct_count, sample_count in zip(topic_correct_counts, topic_sample_counts)]
    return topic_accuracies


df= df.rename(columns = {'remark' : 'text', 'topic':'label'})
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(str)
predict_set = df[df['label']==str(np.nan)]
df = df[df['label']!=str(np.nan)]


topics = df["label"].unique()
train_set = pd.DataFrame()
test_set = pd.DataFrame()
bigrams = pd.DataFrame()
trigrams = pd.DataFrame()
df['text'] = df['text'].apply(lambda x: preprocess_text(x))



for topic in topics:
    # select only the rows with the current topic
    topic_rows = df[df["label"] == topic]
    if len(topic_rows) >= 36:
        topic_rows = topic_rows.sample(n=50)
    # split the topic rows into train and test sets
    train, test = train_test_split(topic_rows, test_size=0.2)

    # add the train and test sets to the overall train and test sets
    train_set = pd.concat([train_set, train])
    test_set = pd.concat([test_set, test])

print("Train set size:", len(train_set))
print("Test set size:", len(test_set))
remarks = df['text'].apply(lambda x: [str(i) for i in x.split(',')]).tolist()

if torch.cuda.is_available():
    # Use GPU for training
    device = torch.device("cuda:0")
else:
    # Use CPU for training
    device = torch.device("cuda:0")
print (device)

print ("downloading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='andrr/setfit_retail')
# tokenizer.padding_side = "left"
# tokenizer.pad_token = tokenizer.eos_token
model = SetFitModel.from_pretrained(pretrained_model_name_or_path='andrr/setfit_retail', labels=2)
torch.cuda.empty_cache()
model.to(device)


# Prepare training data

train_data = Dataset.from_pandas(train_set)
val_data = Dataset.from_pandas(test_set)
test = test_set.copy()

#delete after testing
# new_test = pd.read_csv(fr'C:\Users\yuval\OneDrive\שולחן העבודה\text classification\Updated_train_set 07.11.23.csv', encoding = 'latin-1')
# new_test = new_test.rename(columns = {'topic':'label', 'remark':'text'})
# val_data = Dataset.from_pandas(new_test)
###

early_stopping = EarlyStoppingCallback(early_stopping_patience = 3)

args = TrainingArguments(
    batch_size=6,
    num_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

config = {
    "device": "cuda:0",  # Use the first GPU
    # ... other configuration options
}

trainer = MyTrainer(
    model=model,
    args = args,
    train_dataset=train_data,
    eval_dataset=val_data,
    metric_kwargs={"average": "macro"},
    callbacks=[early_stopping],
    column_mapping={"text": "text", "label": "label"},
    metric =['f1', 'accuracy','precision', 'recall']
)

trainer.train()
metrics = trainer.evaluate(val_data, topics)
print(metrics)



path = r'C:\Users\yuval\OneDrive\שולחן העבודה\text classification\Setfit_text_classification.pth'


model.save_pretrained(fr"C:\Users\yuval\OneDrive\שולחן העבודה\text classification\andrr/setfit_retail")
test_set = pd.read_csv(fr'pd.read_csv(C:\Users\yuval\OneDrive\שולחן העבודה\text classification\preprocessed test set.csv')
predictions = model.predict(test_set['remark'])

# save
def save(model,optimizer):                                                # save model
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

