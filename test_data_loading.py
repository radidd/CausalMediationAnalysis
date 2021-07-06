import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
#from transformers import GPT2Tokenizer

from experiment import Model
from utils import convert_results_to_pd

from UNITER.data import (DetectFeatLmdb, TxtTokLmdb,
                  PrefetchLoader, TokenBucketSampler,
                  Nlvr2PairedEvalDataset, Nlvr2TripletEvalDataset,
                  nlvr2_paired_eval_collate, nlvr2_triplet_eval_collate)

def load_examples(opts, train_opts):
    # TODO-RADI: this is currently copied over from inf_nlvr2.py; load in pairs
    img_db = DetectFeatLmdb(opts.img_db,
                            train_opts.conf_th, train_opts.max_bb,
                            train_opts.min_bb, train_opts.num_bb,
                            opts.compressed_db)
    txt_db = TxtTokLmdb(opts.txt_db, -1)
    dset = EvalDatasetCls(txt_db, img_db, train_opts.use_img_type)
    batch_size = (train_opts.val_batch_size if opts.batch_size is None
                  else opts.batch_size)
    sampler = TokenBucketSampler(dset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=False)
    eval_dataloader = DataLoader(dset, batch_sampler=sampler,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=eval_collate_fn)
    eval_dataloader = PrefetchLoader(eval_dataloader)
    print(eval_dataloader)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        type=str, required=True,
                        help="The input train corpus.")
    parser.add_argument("--img_db",
                        type=str, required=True,
                        help="The input train images.")

    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--batch_size", type=int,
                        help="batch size for evaluation")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--fp16', action='store_true',
                        help="fp16 inference")

    parser.add_argument("--train_dir", type=str, required=True,
                        help="The directory storing NLVR2 finetuning output")
    parser.add_argument("--ckpt", type=int, required=True,
                        help="specify the checkpoint to run inference")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the prediction "
                             "results will be written.")
    args = parser.parse_args()

    train_opts = Struct(json.load(open(f'{args.train_dir}/log/hps.json')))
    loaded_examples = load_examples(opts, train_opts)
