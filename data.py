

import transformers
import torch
import pytorch_lightning as pl
import os.path
import sys
import logging


ID, FORM, LEMMA, UPOS, XPOS, FEAT, HEAD, DEPREL, DEPS, MISC = range(10)

class ConlluData(object):

    def __init__(self):
        self.column_to_label_map = {UPOS: "labels_upos", FEAT: "labels_feat"}
        self.column_to_label_map_inv = {v: k for k,v in self.column_to_label_map.items()}

    def read_conllu(self, f):
        if isinstance(f, str):
            file = open(f, "rt", encoding="utf-8")
        else:
            file = f
        comm = []
        sent = []
        for line in file:
            line = line.strip()
            if not line:
                if sent:
                    yield comm, sent
                comm = []
                sent = []
                continue
            if line.startswith("#"):
                comm.append(line)
                continue
            cols = line.split("\t")
            sent.append(cols)
        else:
            if isinstance(f, str):
                file.close()
            if sent:
                yield comm, sent
                    


    def data2dict(self, f):

        data = []
        for comm, sent in self.read_conllu(f):
            tokens = []
            upos = [] # TODO: add here all you need
            feat = []
            for token in sent:
                tokens.append(token[FORM]) # TODO .replace(" ", "_") needed?
                upos.append(token[UPOS])
                feat.append(token[FEAT])
            data.append({"tokens": tokens, "labels":{"labels_upos": upos, "labels_feat": feat}, "original_data":(comm, sent)})
            
        logging.info(f"{len(data)} sentences in the dataset.")
        return data
        
        
        
    def write_predictions(self, data, predictions, sent_id, output_file):
        
        

        comm, conllu_sent = data[sent_id]["original_data"]
        
        for c in comm:
            print(c, file=output_file)
        for i, token in enumerate(conllu_sent):
            for key in predictions.keys():
                token[self.column_to_label_map_inv[key]] = predictions[key][i]
            print("\t".join(token), file=output_file)
        print(file=output_file)
    
    
        
def collate(data):
    batch = {}
    for k in ["input_ids", "attention_mask", "token_type_ids", "subword_mask"]:
        batch[k] = pad_with_zero([example[k] for example in data])
    batch["labels"] = {}
    for label in data[0]["labels"].keys():
        batch["labels"][label] = pad_with_zero([example["labels"][label] for example in data])
    batch["tokens"] = [example["tokens"] for example in data]
    return batch
    

def pad_with_zero(vals):
    vals = [torch.LongTensor(v) for v in vals]
    padded = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True)
    return padded
            
            

class TaggerDataModule(pl.LightningDataModule):

    def __init__(self, pretrained_bert, label_encoders, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(pretrained_bert)
        self.label_encoders = label_encoders
        
        
    def transform(self, item, inference=False):

        tokens = item["tokens"] # list of tokens
        labels = item["labels"] # dictionary of labels where each label is a list

        
        subwords = [] # list of words where a word is list of subwords
        for token in tokens:
            subwords_ = self.tokenizer.tokenize(token)
            ids_ = self.tokenizer.convert_tokens_to_ids(subwords_)
            subwords.append(ids_)

        subword_labels = {k: [] for k in labels.keys()} # propagate labels for each subword
        subword_mask = [] # first subword of each token is one, rest are masked out (zero)
        for i, word in enumerate(subwords): # i:s word
            for j, subw in enumerate(word): # j:s subword in the i:s word
                for key in subword_labels:
                    subword_labels[key].append(labels[key][i])
                if j == 0:
                    subword_mask.append(1)
                else:
                    subword_mask.append(0)
        subwords = [s for w in subwords for s in w] # flatten the list
 
        assert len(subwords) == len(subword_mask)
        
        encoded = self.tokenizer.prepare_for_model(subwords, add_special_tokens=False, truncation=False)
        encoded_labels = {}
        
        if not inference:
            for key in labels.keys():
                encoded_labels[key] = self.label_encoders[key].transform(subword_labels[key])
        
        d = {"input_ids": encoded.input_ids, "token_type_ids":encoded.token_type_ids, "attention_mask": encoded.attention_mask, "labels": encoded_labels, "subword_mask": subword_mask, "tokens": tokens}

        return d


    def setup(self, data, stage="train"): # stage: train or predict
        
        if stage=="train":
            train_data, eval_data = data
            self.train_data = [self.transform(example) for example in train_data]
            self.eval_data = [self.transform(example) for example in eval_data]
            
        elif stage=="predict":
            self.predict_data = [self.transform(example, inference=True) for example in data]
            
        else:
            raise ValueError(f'Unknown stage {stage} in DataModule.')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, collate_fn=collate, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.eval_data, collate_fn=collate, batch_size=self.batch_size//2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_data, collate_fn=collate, batch_size=self.batch_size)


if __name__=="__main__":

    d = ConlluDataModule("data", 100, "TurkuNLP/bert-base-finnish-cased-v1")
    d.setup()
    dl = d.train_dataloader()
    for x in dl:
        print("_")
    for x in dl:
        print("x")