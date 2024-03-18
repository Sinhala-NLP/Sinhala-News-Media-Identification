import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from experiments.label_encoder import encode, decode
from transformer_model.evaluation import macro_f1, weighted_f1, print_stat
from transformer_model.model_args import ClassificationArgs
from transformer_model.run_model import ClassificationModel
from datasets import Dataset
from datasets import load_dataset

model_type = "roberta"
model_name = "NLPC-UOM/SinBERT-large"

train = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA-Media', split='train'))
test = Dataset.to_pandas(load_dataset('sinhala-nlp/NSINA-Media', split='test'))

train = train.rename(columns={'News Content': 'text', 'Source': 'labels'}).dropna()
train['labels'] = encode(train["labels"])

test = test.rename(columns={'News Content': 'text', 'Source': 'labels'}).dropna()
test['labels'] = encode(test["labels"])

model_args = ClassificationArgs()

model_args.eval_batch_size = 16
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 1000
model_args.evaluate_during_training_verbose = True
model_args.logging_steps = 1000
model_args.learning_rate = 2e-5
model_args.manual_seed = 777
model_args.max_seq_length = 256
model_args.model_type = model_type
model_args.model_name = model_name
model_args.num_train_epochs = 5
model_args.overwrite_output_dir = True
model_args.save_steps = 1000
model_args.train_batch_size = 16
model_args.wandb_project = "NSINa_media_identification"
model_args.regression = False

processed_model_name = model_name.split("/")[1]

model_args.output_dir = os.path.join("outputs", processed_model_name)
model_args.best_model_dir = os.path.join("outputs", processed_model_name, "best_model")
model_args.cache_dir = os.path.join("cache_dir", processed_model_name)


model = ClassificationModel(model_args.model_type, model_args.model_name, num_labels=10, use_cuda=torch.cuda.is_available(),
                     args=model_args)
train, dev = train_test_split(train, test_size=0.1)


model.train_model(train, eval_df=dev, macro_f1=macro_f1, weighted_f1=weighted_f1)
model = ClassificationModel(model_args.model_type, model_args.best_model_dir, num_labels=10,
                     use_cuda=torch.cuda.is_available(), args=model_args)

test_sentences = test['text'].tolist()
predictions, raw_outputs = model.predict(test_sentences)

test['predictions'] = predictions

test['predictions'] = decode(test['predictions'])
test['labels'] = decode(test['labels'])

print_stat(test, 'labels', 'predictions')




