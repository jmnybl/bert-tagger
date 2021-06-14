import torch
import argparse
import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import numpy
from sklearn.preprocessing import LabelEncoder
from data import ConlluData, TaggerDataModule, read_conllu
from model import TaggerModel
import os
import pickle
import logging
from tqdm import tqdm
import time
import sys
import transformers

logging.basicConfig(level=logging.INFO)


def load_model(args):

    # model checkpoint not available
    if not os.path.exists(args.checkpoint_dir):
        logging.error(f"Checkpoint directory {args.checkpoint_dir} does not exists!")
        sys.exit()

    logging.info(f"Loading model from {os.path.join(args.checkpoint_dir, 'best.ckpt')}")
    with open(os.path.join(args.checkpoint_dir, "label_encoders.pickle"), "rb") as f:
       label_encoders = pickle.load(f)
    target_classes = {}
    for label in label_encoders.keys():
        target_classes[label] = len(label_encoders[label].classes_)
    model = TaggerModel.load_from_checkpoint(pretrained_bert=args.bert_pretrained, target_classes=target_classes, checkpoint_path=os.path.join(args.checkpoint_dir, "best.ckpt"))
    model.freeze()
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    return model, label_encoders
    
    
def predict_batch(model, dataset, label_encoders):

    sent_counter = 0
    batch_labels = {}
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset.test_dataloader())):
            predictions = model.predict(batch, i)
            subword_mask = batch["subword_mask"]
            for i, mask in enumerate(subword_mask): # iterate over sentences
                labels = {}
                for key, pred in predictions.items(): # iterate all label sets
                    pred = pred[i][mask != 0]
                    labels[key] = label_encoders[key].inverse_transform(pred.cpu())
                batch_labels[sent_counter] = labels
                sent_counter += 1
    logging.info(f"Predicted {sent_counter} sentences")
    return batch_labels


def data_yielder(fname, batch_size = 500):

    with open(fname, "rt", encoding="utf-8") as f:
        data = f.read()
    batch = []
    for comm, sent in read_conllu(data):
        batch.append((comm, sent))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    else:
        if batch:
            yield batch

def main(args):

    model, label_encoders = load_model(args)
    
    # prepare data objects
    logging.info("Creating data objects")
    datareader = ConlluData()
    
    with open(args.output_file, "wt", encoding="utf-8") as output_file: # open output file
        # read actual data
        for batch in data_yielder(args.data):
            logging.info("Reading data batch")
            data = datareader.data2dict(batch)
            dataset = TaggerDataModule(model.tokenizer, label_encoders, args.batch_size)
            dataset.prepare_data(data, stage="predict")
            dataset.setup("predict")
            batch_labels = predict_batch(model, dataset, label_encoders)
            for sent_idx, labels in batch_labels.items():
                datareader.write_predictions(data, labels, sent_idx, output_file)

    logging.info("Prediction done!")
    
    

    
    






          
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/dev.conllu')
    parser.add_argument('--bert_pretrained', type=str, default='TurkuNLP/bert-base-finnish-cased-v1')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--checkpoint_dir', default="checkpoints", type=str)
    parser.add_argument('--output_file', default="pred.conllu", type=str)
    parser.add_argument('--datareader', default="conllu", type=str) # TODO options
    
    args = parser.parse_args()
    
    main(args)
