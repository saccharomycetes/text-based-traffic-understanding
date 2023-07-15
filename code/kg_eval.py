import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, RobertaTokenizer, RobertaForMaskedLM
from tqdm import tqdm
import json
import jsonlines

skip_words = set(stopwords.words('english'))
PERSON_NAMES = ['Alex', 'Ash', 'Aspen', 'Bali', 'Berkeley', 'Cameron', 'Chris', 'Cody', 'Dana', 'Drew', 'Emory', 'Flynn', 'Gale', 'Jamie', 'Jesse', 
'Kai', 'Kendall', 'Kyle', 'Lee', 'Logan', 'Max', 'Morgan', 'Nico', 'Paris', 'Pat', 'Quinn', 'Ray', 'Robin', 'Rowan', 'Rudy', 'Sam', 'Skylar', 'Sydney', 
'Taylor', 'Tracy', 'West', 'Wynne']

torch.cuda.empty_cache()
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MyDataset(torch.utils.data.Dataset):

	def __init__(self, data, pad_token, mask_token=None, max_words_to_mask=6):
		self.data = data
		self.pad_token = pad_token
		self.mask_token = mask_token
		self.max_words_to_mask = max_words_to_mask

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx] # one sample
		return sample, self.pad_token, self.mask_token, self.max_words_to_mask


class t5_eval():

    def __init__(self, args):
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(args.model_dir, do_lower_case=True)
        count = count_parameters(self.model)
        print ("parameters:"+str(count))
        self.model.to(self.args.device)


    def convert_examples_to_features(self, examples):
        data = []
        for example in examples:

            inputs = ["reasoning:  " + example["question"] + " " + c + "." for c in example["candidates"]]

            input_ids = [self.tokenizer(input, return_tensors='pt',truncation=True).input_ids.numpy().tolist()[0] for input in inputs]
            correct = int(example["answer"])
            candidates = ["1"] * len(example["candidates"])
            theid = example["id"]

            label_id = [self.tokenizer(candidate, return_tensors='pt').input_ids.numpy().tolist()[0] for candidate in candidates]
            data.append([input_ids, label_id, correct, theid])
        return data

    def load_and_tokenize(self):
        with open(self.args.data_dir)as f_dev:
            dev_set = [json.loads(data) for data in f_dev.readlines()]
        dev_data = self.convert_examples_to_features(dev_set)
        return dev_data

    def mCollateFn(self, batch):
        batch_input_ids = []                       # len = batch_size * num_cands
        batch_input_masks = []                     # len = batch_size * num_cands
        batch_labels = []                          # len = batch_size * num_cands
        batch_corrects = [b[0][2] for b in batch]  # len = batch_size
        batch_ids = [b[0][3] for b in batch]

        in_features = [i for b in batch for i in b[0][0]]
        label_features = [i for b in batch for i in b[0][1]]
        pad_token = batch[0][1]
        max_input_len = max([len(f) for f in in_features])
        max_label_len = max([len(f) for f in label_features])
        
        for in_feature, label_feature in zip(in_features, label_features):

            in_sequence = in_feature + [pad_token] * (max_input_len-len(in_feature))
            att_mask = [1] * len(in_feature) + [0] * (max_input_len-len(in_feature))
            label_sequence = label_feature + [pad_token]*(max_label_len-len(label_feature))

            batch_input_ids.append(in_sequence)
            batch_input_masks.append(att_mask)
            batch_labels.append(label_sequence)
        return batch_input_ids, batch_input_masks, batch_labels, batch_corrects, batch_ids
    
    def evaluate(self):

        self.dev_data = self.load_and_tokenize()
        eval_dataset = MyDataset(self.dev_data, self.tokenizer.pad_token_id)

        right_num = 0
        total_num = 0
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, collate_fn=self.mCollateFn)

        # Eval!
        num_can = 0
        rights = []
        preds = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
            self.model.eval()
            with torch.no_grad():
                input_ids = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[0]], dim=0).to(self.args.device)
                att_mask = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[1]], dim=0).to(self.args.device)
                input_labels = torch.cat([torch.LongTensor(c)[0].view(1,-1) for c in batch[2]], dim=0).to(self.args.device)
                outputs = self.model(input_ids = input_ids, attention_mask = att_mask, labels = input_labels)
                num_can = int(input_ids.size(0)/self.args.eval_batch_size)
                logits = outputs[1]
                logits = logits.view(-1, logits.size(-1))
                scores = logits[:,209] - logits[:,204]
                scores = scores.view(-1, num_can)
                answers = torch.argmax(scores, dim=1)
                corrects = torch.LongTensor(batch[3]).to(self.args.device)

                rights.extend([i for i, j in zip(batch[4], answers.eq(corrects)) if j])
                preds.extend([int(a) for a in answers])

                right_num += int(torch.sum(answers.eq(corrects)))
                total_num += len(batch[3])
                
        output_acc_file = os.path.join(self.args.output_dir, "acc.txt")
        with open(output_acc_file, "w") as writer:
            print("***** Eval results *****")
            print("  acc = %s", str(right_num/total_num))
            writer.write("acc = %s\n" % (str(right_num/total_num)))
        preds_info = {"preds" : preds, "rights": rights}
        output_preds_file = os.path.join(self.args.output_dir, "preds.json")
        with open(output_preds_file, "w") as f:
            json.dump(preds_info ,f)
        return right_num/total_num
    
class roberta_eval():

    def __init__(self, args):
        self.args = args
        self.model = RobertaForMaskedLM.from_pretrained(args.model_dir)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_dir, do_lower_case=True)
        count = count_parameters(self.model)
        print ("parameters:"+str(count))
        self.model.to(self.args.device)
    
    def handle_words(self, span, keywords=None, is_start=False):
        inputs = []
        labels = []
        words = nltk.word_tokenize(span)
        for w_i, w in enumerate(words):
            if (w_i == 0 and is_start) or w == '.' or w == ',' or w.startswith('\''):
                w_bpes = self.tokenizer.tokenize(w)
            else:
                w_bpes = self.tokenizer.tokenize(w, add_prefix_space=True)
            inputs.extend(w_bpes)
            if keywords != None:
                if w in keywords:
                    labels.extend(w_bpes)
                else:
                    labels.extend([-100]*len(w_bpes))
            else:
                if w not in PERSON_NAMES and w not in skip_words and w.lower() not in skip_words:
                    labels.extend(w_bpes)
                else:
                    labels.extend([-100]*len(w_bpes))
        return inputs, labels

    def handle_underscores(self, suffix, keywords=None, prefix=False):
        inputs = []
        labels = []
        if '_' in suffix:
            suffix_parts = [i.strip() for i in suffix.split('___')]
            for i, part in enumerate(suffix_parts):
                if part:
                    tmp_inputs, tmp_labels = self.handle_words(part, keywords=keywords, is_start=(i==0 and prefix))
                    inputs += tmp_inputs
                    labels += tmp_labels

                    if i != len(suffix_parts) - 1 and suffix_parts[i+1]:
                        inputs.append(self.tokenizer.mask_token)
                        labels.append(-100) # -100 => skip words?
                else:
                    inputs.append(self.tokenizer.mask_token)
                    labels.append(-100)
        else:
            inputs, labels = self.handle_words(suffix, keywords=keywords, is_start=prefix)
        return inputs, labels

    def convert_examples_to_features(self, examples):
        data = []
        for example in examples:
            con = example['question']
            cands = example['candidates']
            inputs, labels = self.handle_underscores(con, prefix=True)
            choices = [self.handle_underscores(cand) for cand in cands]
            input_ids = [inputs+cand[0] for cand in choices]
            input_ids = [self.tokenizer.convert_tokens_to_ids(cand) for cand in input_ids]
            label_ids = [labels+cand[1] for cand in choices]
            
            label_ids = [[t if t == -100 else input_ids[i][t_i] for t_i, t in enumerate(cand)] for i, cand in enumerate(label_ids)]
            label_ids = [[-100]+cand+[-100] for cand in label_ids]
            input_ids = [self.tokenizer.prepare_for_model(cand, max_length=512, truncation=True)['input_ids'] for cand in input_ids]

            dimension = self.args.data_dir
            sample_id = example["id"]
            data.append([input_ids, label_ids, int(example['answer']), dimension, sample_id])	
                    
        return data

    def load_and_tokenize(self):
        with open(self.args.data_dir)as f_dev:
            dev_set = [json.loads(data) for data in f_dev.readlines()]
        dev_data = self.convert_examples_to_features(dev_set)
        return dev_data

    def mCollateFn(self, batch):
        batch_input_ids = []
        batch_input_mask = []
        batch_input_labels = []
        batch_label_ids = []
        features = [b[0] for b in batch]
        pad_token = batch[0][1]
        mask_token = batch[0][2]
        MAX_WORDS_TO_MASK = batch[0][3]
        max_len = max([len(cand) for f in features for cand in f[0]])

        batch_dimensions = []
        batch_ids = []

        for f in features:
            batch_input_ids.append([])
            batch_input_mask.append([])
            batch_input_labels.append([])
            batch_label_ids.append(f[2])
            batch_dimensions.append(f[3]) # dimension
            batch_ids.append(f[4]) # sample id

            for i in range(len(f[0])):
                masked_sequences = []
                masked_labels = []
                this_att_mask = []
                sequence = f[0][i] + [pad_token]*(max_len-len(f[0][i]))
                label_sequence = f[1][i]+[-100]*(max_len-len(f[1][i]))
                valid_indices = [l_i for l_i, l in enumerate(label_sequence) if l != -100]
                if len(valid_indices) > MAX_WORDS_TO_MASK:
                    rm_indices = random.sample(valid_indices, (len(valid_indices)-MAX_WORDS_TO_MASK))
                    label_sequence = [-100 if l_i in rm_indices else l for l_i, l in enumerate(label_sequence)]
                for j, t in enumerate(label_sequence):
                    if t == -100:
                        continue
                        masked_sequences.append(sequence)
                        masked_labels.append([-100]*max_len)
                    else:
                        masked_sequences.append(sequence[:j]+[mask_token]+sequence[j+1:])
                        masked_labels.append([-100]*j+[sequence[j]]+[-100]*(max_len-j-1))
                    this_att_mask.append([1]*len(f[0][i])+[0]*(max_len-len(f[0][i])))
                batch_input_ids[-1].append(torch.tensor(masked_sequences, dtype=torch.long))
                batch_input_mask[-1].append(torch.tensor(this_att_mask, dtype=torch.long))
                batch_input_labels[-1].append(torch.tensor(masked_labels, dtype=torch.long))
            
            
        return batch_input_ids, batch_input_mask, batch_input_labels, torch.tensor(batch_label_ids, dtype=torch.long),batch_dimensions,batch_ids

    def evaluate(self):
        results = {}
        self.args.max_sequence_per_time = 80
        self.dev_data = self.load_and_tokenize()
        eval_dataset = MyDataset(self.dev_data, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, 100)
        
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, collate_fn=self.mCollateFn)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        # Eval!
        right_num = 0
        total_num = 0
        rights = []
        preds = []
		
        CE = torch.nn.CrossEntropyLoss(reduction='none')
        out_label_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
            self.model.eval()
            with torch.no_grad():
                num_cand = len(batch[0][0])
                choice_loss = []
                choice_seq_lens = np.array([0]+[len(c) for sample in batch[0] for c in sample])
                choice_seq_lens = np.cumsum(choice_seq_lens)
                input_ids = torch.cat([c for sample in batch[0] for c in sample], dim=0).to(self.args.device)
                att_mask = torch.cat([c for sample in batch[1] for c in sample], dim=0).to(self.args.device)
                input_labels = torch.cat([c for sample in batch[2] for c in sample], dim=0).to(self.args.device)
                if len(input_ids) < self.args.max_sequence_per_time:
                    inputs = {'input_ids': input_ids,
                            'attention_mask': att_mask}
                    outputs = self.model(**inputs)
                    ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels.view(-1))
                    ce_loss = ce_loss.view(outputs[0].size(0), -1).sum(1)
                else:
                    ce_loss = []
                    for chunk in range(0, len(input_ids), self.args.max_sequence_per_time):
                        inputs = {'input_ids': input_ids[chunk:chunk+self.args.max_sequence_per_time],
                            'attention_mask': att_mask[chunk:chunk+self.args.max_sequence_per_time]}
                        outputs = self.model(**inputs)
                        tmp_ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels[chunk:chunk+self.args.max_sequence_per_time].view(-1))
                        tmp_ce_loss = tmp_ce_loss.view(outputs[0].size(0), -1).sum(1)
                        ce_loss.append(tmp_ce_loss)
                    ce_loss = torch.cat(ce_loss, dim=0)
                for c_i in range(len(choice_seq_lens)-1):
                    start = choice_seq_lens[c_i]
                    end =  choice_seq_lens[c_i+1]
                    choice_loss.append(-ce_loss[start:end].sum()/(end-start))
                choice_loss = torch.stack(choice_loss)
                breakpoint()
                choice_loss = choice_loss.view(-1, num_cand)
                answers = torch.argmax(choice_loss, dim=1)
                corrects = torch.LongTensor(batch[3]).to(self.args.device)
                rights.extend([i for i, j in zip(batch[5], answers.eq(corrects)) if j])
                preds.extend([int(a) for a in answers])
                right_num += int(torch.sum(answers.eq(corrects)))
                total_num += len(batch[3])

        output_acc_file = os.path.join(self.args.output_dir, "acc.txt")
        with open(output_acc_file, "w") as writer:
            print("***** Eval results *****")
            print("  acc = %s", str(right_num/total_num))
            writer.write("acc = %s\n" % (str(right_num/total_num)))
        preds_info = {"preds" : preds, "rights": rights}
        output_preds_file = os.path.join(self.args.output_dir, "preds.json")
        with open(output_preds_file, "w") as f:
            json.dump(preds_info ,f)
        return right_num/total_num


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", default="./", type=str,
                        help="The data name")
	parser.add_argument("--output_dir", default="../result/", type=str,
                        help="The data name")
	parser.add_argument("--model_dir", default="../model/roberta_large_kg", type=str,
                        help="The data name")
	parser.add_argument("--eval_batch_size", default=8, type=int,
						help="Batch size per GPU/CPU for evaluation.")
	args = parser.parse_args()


	free_gpu = get_freer_gpu()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	torch.cuda.set_device(free_gpu)

	args.device = device

	model = t5_eval(args) if args.model_dir.lower().find("t5") != -1 else roberta_eval(args)

	result = model.evaluate()

	return result

if __name__ == "__main__":
	main()