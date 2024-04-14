import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from pytorch_lightning import seed_everything
from utils import read_json, AverageMeterSet, Ranker
from dataloader import RecformerEvalDataset
import time
from typing import Dict, Union, List
from dataclasses import dataclass
import numpy as np
from sentence_transformers.util import cos_sim
import mistral_ollama

def load_data(args):

    train = read_json(
        os.path.join(args.data_path, args.train_file), True
    )  # Dict: {user -> [item1 id, item2 id, ...], ...}
    val = read_json(
        os.path.join(args.data_path, args.dev_file), True
    )  # Dict: {user -> [item1 id, item2 id, ...], ...}
    test = read_json(
        os.path.join(args.data_path, args.test_file), True
    )  # Dict: {user -> [item1 id, item2 id, ...], ...}
    item_meta_dict = json.load(
        open(os.path.join(args.data_path, args.meta_file))
    )  # Dict: {item -> {'title': x, 'brand': y, 'category': z}, ...}

    item2id = read_json(
        os.path.join(args.data_path, args.item2id_file)
    )  # Dict: {item -> id, ...}
    id2item = {v: k for k, v in item2id.items()}

    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v

    return train, val, test, item_meta_dict_filted, item2id, id2item


def _par_tokenize_doc(doc, args):

    item, item_attr = doc

    if args.no_prompt:
        inputs = f"""{str(item_attr)}"""
        flatten_tokens = tokenizer.tokenize(inputs)
    else:
        inputs = f"""Item Title: {item_attr['title']} Brand: {item_attr['brand']} Category: {item_attr['category']}"""
        flatten_tokens = tokenizer.tokenize(inputs)

    return item, flatten_tokens


@dataclass
class EvalDataCollatorWithPadding:

    tokenizer: AutoTokenizer
    tokenized_items: Dict
    args: ArgumentParser

    def __call__(
        self,
        batch_data: List[
            Dict[str, Union[int, List[int], List[List[int]], torch.Tensor]]
        ],
    ) -> Dict[str, torch.Tensor]:
        item_idss = [ele["items"] for ele in batch_data]
        labels = [ele["label"] for ele in batch_data] if "label" in batch_data[0] else None
        item_idss = [item_ids[::-1] for item_ids in item_idss]  # reverse items order
        item_idss = [item_ids[:50] for item_ids in item_idss]  # truncate the number of items
        if self.args.no_prompt:
            tokenss = [[token for id in item_ids for token in self.tokenized_items[id]] for item_ids in item_idss]
            batch = self.tokenizer(
                tokenss,
                padding="longest",
                return_tensors="pt",
                is_split_into_words=True,
                truncation=True,
                max_length=1024,
            )
        else:
            prefix_tokens = tokenizer.tokenize('Item Sequence is as follows: ')
            tokenss = [prefix_tokens + [token for id in item_ids for token in self.tokenized_items[id]] for item_ids in item_idss]
            batch = self.tokenizer(
                tokenss,
                padding="longest",
                return_tensors="pt",
                is_split_into_words=True,
            )
        if labels:
            labels = torch.tensor(labels)

        return batch, labels

def encode_all_items(tokenized_items, collator, args):

    class AllItemDataset(Dataset):
        def __init__(
            self,
            tokenized_items,
        ):
            self.tokenized_items = tokenized_items

        def __len__(self):
            return len(self.tokenized_items)

        def __getitem__(self, index):
            return self.tokenized_items[index]

        def collate_fn(self, data):
            return tokenizer(data, padding="longest", return_tensors="pt", is_split_into_words=True)


    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []
    dataset = AllItemDataset(items)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    with torch.no_grad():
        for batch in tqdm(loader, ncols=100, desc="Encode all items"):

            inputs = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**inputs)
            item_embeddings.append(outputs.last_hidden_state[:, -1].detach().cpu())

    item_embeddings = torch.cat(item_embeddings, dim=0)
    return item_embeddings

# For retrieval you need to pass this prompt. Please find our more in our blog post.
def transform_query(query: str) -> str:
    """For retrieval, add the prompt for query (not for documents)."""
    return f"Represent this sentence for searching relevant passages: {query}"


# The model works really well with cls pooling (default) but also with mean poolin.
def pooling(outputs: torch.Tensor, inputs: Dict, strategy: str = "last") -> torch.Tensor:
    if strategy == "last":
        # take the last hidden states in where the attention mask is 1
        outputs = outputs[:, -1]
    elif strategy == "mean":
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1
        ) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    return outputs.detach().cpu()#.numpy()

def paged_cos_sim(batch_embeddings: Tensor, item_embeddings: Tensor, args) -> Tensor:
    """
    Cosine similarity between embeddings and item_embeddings.
    """
    batch_size = args.batch_size
    device = args.device
    batch_embeddings = batch_embeddings.unsqueeze(1).to(device) # (batch_size, 1, hidden_size)
    similarities = [
        F.cosine_similarity(
            batch_embeddings, 
            item_embeddings[i : i + batch_size].to(device).unsqueeze(0), # (1, batch_size, hidden_size)
            dim=-1
        ).cpu()
        for i in range(0, len(item_embeddings), batch_size)
    ] # list of (batch_size, batch_size)
    return torch.cat(similarities, dim=1) # (batch_size, num_item)

class PagedRanker(nn.Module):
    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        
    def forward(self, scores, labels):
        labels = labels.squeeze()
        
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        
        valid_length = (scores > -1e4).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        res.append((1 - (rank/valid_length)).mean().item()) # AUC

        return res + [None]

# 1. load model
model_id = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).cuda().eval()
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
model.config.pad_token_id = tokenizer.eos_token_id

def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument("--data_path", type=str, default=None, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt", type=str, default="best_model.bin")
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--dev_file", type=str, default="val.json")
    parser.add_argument("--test_file", type=str, default="test.json")
    parser.add_argument("--item2id_file", type=str, default="smap.json")
    parser.add_argument("--meta_file", type=str, default="meta_data.json")

    # data process
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # model
    parser.add_argument(
        "--temp", type=float, default=0.05, help="Temperature for softmax."
    )

    # train
    parser.add_argument(
        "--metric_ks", nargs="+", type=int, default=[10, 50], help="ks for Metric@k"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_prompt", action="store_true")

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    start_time = time.time()
    args.device = (
        torch.device("cuda:{}".format(args.device))
        if args.device >= 0
        else torch.device("cpu")
    )

    train, val, test, item_meta_dict, item2id, id2item = load_data(args)

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / "preprocess"
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / (f"tokenized_items_{path_corpus.name}_mistral" if args.no_prompt else f"tokenized_items_{path_corpus.name}_prompt")

    if path_tokenized_items.exists():
        print(f"[Preprocessor] Use cache: {path_tokenized_items}")
    else:
        print(f"Loading attribute data {path_corpus}")

        doc_tuples = [_par_tokenize_doc(ele, args) for ele in item_meta_dict.items()]
        tokenized_items = {item2id[item]: tokens for item, tokens in doc_tuples}

        torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f"Successfully load {len(tokenized_items)} tokenized items.")

    collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items, args)
    data = RecformerEvalDataset(train, val, test, mode="test", collator=collator)
    loader = DataLoader(
        data, batch_size=args.batch_size, collate_fn=data.collate_fn
    )

    path_item_embeddings = dir_preprocess / (f"item_embeddings_{path_corpus.name}_mistral" if args.no_prompt else f"item_embeddings_{path_corpus.name}_prompt")
    if path_item_embeddings.exists():
        print(f"[Item Embeddings] Use cache: {path_tokenized_items}")
    else:
        print(f"Encoding items.")
        item_embeddings = encode_all_items(tokenized_items, collator, args)
        torch.save(item_embeddings, path_item_embeddings)

    item_embeddings = torch.load(path_item_embeddings)
    ranker = PagedRanker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    for batch in tqdm(loader, ncols=100, desc="Evaluate"):

        inputs = {k: v.to(args.device) for k, v in batch[0].items()}
        labels = batch[1] if len(batch) == 2 else None

        with torch.no_grad():
            # with autocast(enabled=args.fp16):
            outputs = model(**inputs).last_hidden_state
            embeddings = pooling(outputs, inputs, "last")
            similarities = paged_cos_sim(embeddings, item_embeddings, args)
            res = ranker(similarities, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics["NDCG@%d" % k] = res[2 * i]
            metrics["Recall@%d" % k] = res[2 * i + 1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

    average_metrics = average_meter_set.averages()

    print(f"Test set: {average_metrics}")
    print(f"Total Hours: {(time.time() - start_time) / 3600} hrs")


if __name__ == "__main__":
    main()
