from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json


torch.cuda.empty_cache()
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id

class MyDataset(torch.utils.data.Dataset):

	def __init__(self, data, pad_token):
		self.data = data
		self.pad_token = pad_token

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx] # one sample
		return sample, self.pad_token

def load_and_tokenize(args, tokenizer):
	with open(args.data_dir)as f_dev:
		dev_set = [json.loads(data) for data in f_dev.readlines()]
	dev_data = convert_examples_to_features(args, dev_set, tokenizer)
	return dev_data

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert_examples_to_features(args, examples, tokenizer):
    data = []
    for i, example in tqdm(enumerate(examples),ncols=100):
        ids = [tokenizer.encode(example["question"], c + ".", return_tensors='pt', truncation=True).numpy().tolist()[0] for c in example["candidates"]]
        answer = int(example["answer"])
        data_id = example["id"]
        data.append([ids, answer, data_id])
    return data

def mCollateFn(batch):
	batch_input_ids = []                       # len = batch_size * num_cands
	batch_input_masks = []                     # len = batch_size * num_cands
	batch_corrects = [b[0][1] for b in batch]  # len = batch_size
	batch_ids = [b[0][2] for b in batch]

	in_features = [c for b in batch for c in b[0][0]]
	pad_token = batch[0][1]
	max_input_len = max([len(f) for f in in_features])

	for in_feature in in_features:

		in_sequence = in_feature + [pad_token]*(max_input_len-len(in_feature))
		att_mask = [1] * len(in_feature) + [0] * (max_input_len-len(in_feature))
		batch_input_ids.append(in_sequence)
		batch_input_masks.append(att_mask)

	return batch_input_ids, batch_input_masks, batch_corrects, batch_ids

def eval(args, model, eval_dataset):
	right_num = 0
	total_num = 0
	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	# Note that DistributedSampler samples randomly
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=mCollateFn)
	num_can = 0
	rights = []
	preds = []

	for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=100):
		model.eval()
		with torch.no_grad():
			input_ids = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[0]], dim=0).to(args.device)
			att_mask = torch.cat([torch.LongTensor(c).view(1,-1) for c in batch[1]], dim=0).to(args.device)
			logits = model(input_ids = input_ids, attention_mask = att_mask)["logits"]
			e_logits = logits[:,[0,2]]
			num_can = int(input_ids.size(0)/args.eval_batch_size)
			probs = e_logits.softmax(dim=1)
			scores = probs[:,1]
			scores = scores.view(-1, num_can)
			answers = torch.argmax(scores, dim=1)
			corrects = torch.LongTensor(batch[2]).to(args.device)
			right_num += int(torch.sum(answers.eq(corrects)))
			total_num += len(batch[3])
			rights.extend([b for b, e in zip(batch[3], answers.eq(corrects)) if e])
			preds.extend([int(a) for a in answers])
	output_acc_file = os.path.join(args.output_dir, "acc.txt")

	with open(output_acc_file, "w") as writer:
		print("***** Eval results *****")
		print("model:" + args.model_dir)
		print("  acc = %s", str(right_num/total_num))
		writer.write("acc = %s\n" % (str(right_num/total_num)))
	
	preds_info = {"preds" : preds, "rights": rights}
	output_preds_file = os.path.join(args.output_dir, "preds.json")
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
	model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
	tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
	
	dev_data = load_and_tokenize(args, tokenizer)
	eval_dataset = MyDataset(dev_data, tokenizer.pad_token_id)

	count = count_parameters(model)
	print ("model parameters:"+str(count))
	model.eval()
	model.to(args.device)
	result = eval(args, model, eval_dataset)
	return result


if __name__ == "__main__":
	main()