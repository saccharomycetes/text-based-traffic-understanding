from torch.cuda import reset_max_memory_allocated
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer, util
import os
import numpy as np
import json
from tqdm import tqdm
import argparse
import difflib
import random

def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

def find_similarity(small, large_list):
    similarity_list = []
    for i, large in enumerate(large_list):
        max_similarity = 0
        for j in range(len(large) - len(small) + 1):
            similarity = sum([1 for a, b in zip(small, large[j:j + len(small)]) if a == b])
            if similarity > max_similarity:
                max_similarity = similarity
        similarity_list.append((i, max_similarity))
    return [x[0] for x in sorted(similarity_list, key=lambda x: x[1], reverse=True)]



def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.cuda.empty_cache()
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_available = [-int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system("rm tmp")
    GPU_id = int(np.argmax(memory_available))
    print("using GPU{}".format(GPU_id))
    return GPU_id

class dpr_uni_model():

    def __init__(self, args):
        self.args = args
        self.tokenizer = T5Tokenizer.from_pretrained(self.args.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.args.model_dir)
        self.passage_encoder = SentenceTransformer('facebook-dpr-ctx_encoder-single-nq-base')
        self.query_encoder = SentenceTransformer('facebook-dpr-question_encoder-single-nq-base')
        count = count_parameters(self.model)
        print (f"QA model parameters:{count}")
        self.model.to(self.args.device)
    
    def construct_input(self, question, candidates, sort_index, passages):
        input_string = question + "\\n"
        options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        for i, ans in enumerate(candidates):
            input_string += " ("+options[i]+") "+ ans
        if self.args.num_related > 0:
            for i in sort_index[:self.args.num_related]:
                input_string += "\\n "
                input_string += passages[i]
        input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.args.device)
        return input_id

    def load_data(self):
        with open(self.args.data_dir)as f:
            datas = [json.loads(line) for line in f.readlines()]
        if self.args.num_related > 0:
            with open(self.args.corpus_file)as f:
                paras = f.readlines()
                paras = [json.loads(line) for line in paras]
            passages = [i['text'] for i in paras]
            # passages = [i for j in paras.keys() for k in paras[j].keys() for i in paras[j][k]]
        else:
            passages = []
        return datas, passages
    
    def evaluate(self):
        datas, passages = self.load_data()
        if self.args.num_related > 0:
            passage_embeddings = self.passage_encoder.encode(passages, batch_size=32, show_progress_bar=True)

        pred_answers = []
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        preds = []
        rights = []
        for data in tqdm(datas, ncols=100):
            if self.args.num_related > 0:
                # sort_index = find_similarity(data["question"], passages)
                q_embedding = self.query_encoder.encode(data["question"])
                related_scores = util.dot_score(q_embedding, passage_embeddings)
                sort_index = torch.sort(related_scores, dim=1, descending=True)[1][0].tolist()
            else:
                sort_index = []

            input_id = self.construct_input(data["question"], data["candidates"], sort_index, passages)
            pred_id = self.model.generate(input_id)
            pred_answer = self.tokenizer.batch_decode(pred_id, skip_special_tokens=True)[0]
            similaritiess = np.array([string_similar(pred_answer, k) for k in data["candidates"]])
            pred_idx = int(np.argmax(similaritiess))
            preds.append(pred_idx)
            if pred_idx == int(data["answer"]):
                rights.append(data["id"])

        # similaritiess = np.array([[string_similar(i, k) for k in j["candidates"]] for i, j in zip(pred_answers, datas)], dtype=object)

        # preds = [int(i) for i in (np.argmax(similaritiess, axis=1))]
        # corrects = [int(data["answer"]) for data in datas]
        # ids = [data["id"] for data in datas]

        # rights = [k for i, j, k in zip(preds, corrects, ids) if i==j]
        acc = len(rights)/len(preds)

        output_acc_file = os.path.join(self.args.output_dir, "acc.txt")
        with open(output_acc_file, "w") as writer:
            print("***** Eval results *****")
            print("pas_num:" + str(self.args.num_related))
            print("  acc = %s", str(acc))
            writer.write("acc = %s\n" % (str(acc)))
        preds_info = {"preds": preds, "rights": rights}
        output_preds_file = os.path.join(self.args.output_dir, "preds.json")
        with open(output_preds_file, "w") as f:
            json.dump(preds_info ,f)
        return acc

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./", type=str,
                        help="The data name")
    parser.add_argument("--output_dir", default="../result/", type=str,
                        help="The data name")
    parser.add_argument("--num_related", default=0, type=int,
                        help="The data name")
    parser.add_argument("--corpus_file", default="../result/", type=str,
                        help="The data name")
    parser.add_argument("--model_dir", default="allenai/unifiedqa-v2-t5-large-1251000", type=str,
                        help="The data name")
    parser.add_argument("--eval_batch_size", default=8, type=int,
						help="Batch size per GPU/CPU for evaluation.")

    args = parser.parse_args()


    free_gpu = get_freer_gpu()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(free_gpu)
    args.device = device

    model = dpr_uni_model(args)

    result = model.evaluate()
    return result


if __name__ == "__main__":
	main()