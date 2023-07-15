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
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import json
import jsonlines
logger = logging.getLogger(__name__)

skip_words = set(stopwords.words('english'))
PERSON_NAMES = ['Alex', 'Ash', 'Aspen', 'Bali', 'Berkeley', 'Cameron', 'Chris', 'Cody', 'Dana', 'Drew', 'Emory', 'Flynn', 'Gale', 'Jamie', 'Jesse', 
'Kai', 'Kendall', 'Kyle', 'Lee', 'Logan', 'Max', 'Morgan', 'Nico', 'Paris', 'Pat', 'Quinn', 'Ray', 'Robin', 'Rowan', 'Rudy', 'Sam', 'Skylar', 'Sydney', 
'Taylor', 'Tracy', 'West', 'Wynne']

torch.cuda.empty_cache()
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))

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

    
class roberta_train():

    def __init__(self, args):
        self.args = args
        self.model = RobertaForMaskedLM.from_pretrained(args.model)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=True)
        count = count_parameters(self.model)
        print("parameters:"+str(count))
        self.model.to(self.args.device)
        self.dev_data = self.load_and_tokenize_dev()
        self.eval_dataset = MyDataset(self.dev_data, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, 100)
    
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
        for example in tqdm(examples):

            cands = [example["question"] + " " + c for c in example["candidates"]]

            choices = [self.handle_underscores(cand) for cand in cands]
            input_ids = [cand[0] for cand in choices]
            input_ids = [self.tokenizer.convert_tokens_to_ids(cand) for cand in input_ids]
            label_ids = [cand[1] for cand in choices]
            
            label_ids = [[t if t == -100 else input_ids[i][t_i] for t_i, t in enumerate(cand)] for i, cand in enumerate(label_ids)]
            label_ids = [[-100]+cand+[-100] for cand in label_ids]
            input_ids = [self.tokenizer.prepare_for_model(cand, max_length=512, truncation=True)['input_ids'] for cand in input_ids]

            dimension = self.args.train_file
            sample_id = example["id"]
            data.append([input_ids, label_ids, int(example['answer']), dimension, sample_id])
                    
        return data

    def load_and_tokenize_dev(self):
        with open(self.args.test_file)as f_dev:
            dev_set = [json.loads(data) for data in f_dev.readlines()]
        dev_data = self.convert_examples_to_features(dev_set)
        return dev_data
    
    def load_and_tokenize_train(self):
        with open(self.args.train_file)as f_train:
            train_set = [json.loads(data) for data in f_train.readlines()]
        train_data = self.convert_examples_to_features(train_set)
        return train_data

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
            
        return batch_input_ids, batch_input_mask, batch_input_labels, torch.tensor(batch_label_ids, dtype=torch.long), batch_dimensions, batch_ids

    def train(self):

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Setup logging
        log_file = os.path.join(self.args.output_dir, 'train.log')
        logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt = '%m/%d/%Y %H:%M:%S',
                            level = logging.INFO ,
                            filename=log_file)
        os.system("cp roberta_train.py %s" % os.path.join(self.args.output_dir, 'roberta_train.py'))
        
        self.train_data = self.load_and_tokenize_train()
        train_dataset = MyDataset(self.train_data, self.tokenizer.pad_token_id, self.tokenizer.mask_token_id, 6)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, collate_fn=self.mCollateFn)


        t_total = len(train_dataloader) // self.args.acc_step * self.args.num_epoches


        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        warmup_steps = int(0.05 * t_total)
        logger.info("warm up steps = %d", warmup_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-6, betas=(0.9, 0.98))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_epoches)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    self.args.train_batch_size * self.args.acc_step)
        logger.info("  Gradient Accumulation steps = %d", self.args.acc_step)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        train_iterator = trange(int(self.args.num_epoches), desc="Epoch", ncols=100)

        curr_best = 0.0
        CE = torch.nn.CrossEntropyLoss(reduction='none')
        loss_fct = torch.nn.MultiMarginLoss(margin=1)
        epoch = 0
        for _ in train_iterator: # how many epochs does this model do training
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", ncols=100)
            
            for step, batch in enumerate(epoch_iterator):
                # len(batch) = 5
                self.model.train()
                num_cand = len(batch[0][0]) # 3 answers
                choice_loss = []
                choice_seq_lens = np.array([0]+[len(c) for sample in batch[0] for c in sample])
                choice_seq_lens = np.cumsum(choice_seq_lens)
                input_ids = torch.cat([c for sample in batch[0] for c in sample], dim=0).to(self.args.device)
                att_mask = torch.cat([c for sample in batch[1] for c in sample], dim=0).to(self.args.device)
                input_labels = torch.cat([c for sample in batch[2] for c in sample], dim=0).to(self.args.device)

                if len(input_ids) < 200:
                    inputs = {'input_ids': input_ids,
                            'attention_mask': att_mask}
                    outputs = self.model(**inputs)
                    ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels.view(-1))
                    ce_loss = ce_loss.view(outputs[0].size(0), -1).sum(1)
                else:
                    ce_loss = []
                    for chunk in range(0, len(input_ids), 200):
                        inputs = {'input_ids': input_ids[chunk:chunk+200],
                            'attention_mask': att_mask[chunk:chunk+200]}
                        outputs = self.model(**inputs)
                        tmp_ce_loss = CE(outputs[0].view(-1, outputs[0].size(-1)), input_labels[chunk:chunk+200].view(-1))
                        tmp_ce_loss = tmp_ce_loss.view(outputs[0].size(0), -1).sum(1)
                        ce_loss.append(tmp_ce_loss)
                    ce_loss = torch.cat(ce_loss, dim=0)
                # all tokens are valid
                for c_i in range(len(choice_seq_lens)-1):
                    start = choice_seq_lens[c_i]
                    end =  choice_seq_lens[c_i+1]
                    choice_loss.append(-ce_loss[start:end].sum()/(end-start))

                choice_loss = torch.stack(choice_loss)
                choice_loss = choice_loss.view(-1, num_cand)
                loss = loss_fct(choice_loss, batch[3].to(self.args.device))

                if self.args.acc_step > 1:
                    loss = loss / self.args.acc_step

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.acc_step == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:

                        logger.info(" global_step = %s, average loss = %s", global_step, (tr_loss - logging_loss)/self.args.logging_steps)
                        logging_loss = tr_loss

            # get the best acc
            acc = self.evaluate()
            if acc > curr_best:
                curr_best = acc
                # Save model checkpoint
                output_dir = self.args.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_save.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

                # save the best model under checkpoints for each epoch
                epoch_check_dir = os.path.join(self.args.output_dir, 'checkpoint', str(epoch))
                if not os.path.exists(epoch_check_dir):
                    os.makedirs(epoch_check_dir)
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_save.save_pretrained(epoch_check_dir)
                self.tokenizer.save_pretrained(epoch_check_dir)
                torch.save(self.args, os.path.join(epoch_check_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", epoch_check_dir)
                        
            epoch+=1

        acc = self.evaluate()

        if acc > curr_best:
            curr_best = acc
            # Save model checkpoint
            output_dir = self.args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        return global_step, tr_loss / global_step, curr_best

    def evaluate(self):
        results = {}
        
        eval_sampler = SequentialSampler(self.eval_dataset)
        eval_dataloader = DataLoader(self.eval_dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size, collate_fn=self.mCollateFn)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(self.eval_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

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
                choice_loss = choice_loss.view(-1, num_cand)
                answers = torch.argmax(choice_loss, dim=1)
                corrects = torch.LongTensor(batch[3]).to(self.args.device)
                rights.extend([i for i, j in zip(batch[4], answers.eq(corrects)) if j])
                preds.extend([int(a) for a in answers])
                right_num += int(torch.sum(answers.eq(corrects)))
                total_num += len(batch[3])

        output_acc_file = os.path.join(self.args.output_dir, "acc.txt")
        with open(output_acc_file, "a") as writer:
            print("  acc = %s", str(right_num/total_num))
            logger.info("  acc = %s", str(right_num/total_num))
            writer.write("acc = %s\n" % (str(right_num/total_num)))
        preds_info = {"preds" : preds, "rights": rights}
        output_preds_file = os.path.join(self.args.output_dir, "preds.json")
        with open(output_preds_file, "a") as f:
            f.write(json.dumps(preds_info))
            f.write("\n")
        return right_num/total_num


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="../data/train.jsonl", type=str, 
                        help="The train data name")
    parser.add_argument("--test_file", default="../data/test.jsonl", type=str, 
                        help="The test data name")
    parser.add_argument("--output_dir", default="../model/roberta1", type=str, 
                        help="The test data name")
    parser.add_argument("--max_sequence_per_time", default=200, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--acc_step", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="train batch size")
    parser.add_argument("--num_epoches", default=5, type=int,
                        help="number of epoches to train")
    parser.add_argument("--model", default="roberta-large", type=str,
                        help="name or path of model")
    args = parser.parse_args()
    
    args.logging_steps = 10
    args.max_sequence_per_time = 200

    free_gpu = get_freer_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(free_gpu)

    args.device = device

    model = roberta_train(args)

    global_step, tr_loss, best_acc = model.train()
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    return best_acc

if __name__ == "__main__":
	main()