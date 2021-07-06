"""Run all the extraction for a model across many templates.
"""
# add UNITER to directories to search for python packages
import sys
sys.path.append('./UNITER/')

import argparse
import os
from datetime import datetime
import json

import torch
from torch.utils.data import DataLoader
#from transformers import GPT2Tokenizer

from experiment import Model
from utils_cma import convert_results_to_pd

from horovod import torch as hvd

from UNITER.data import (DetectFeatLmdb, TxtTokLmdb,
                  PrefetchLoader, TokenBucketSampler,
                  Nlvr2PairedEvalDataset, Nlvr2TripletEvalDataset,
                  nlvr2_paired_eval_collate, nlvr2_triplet_eval_collate)

from UNITER.utils.misc import Struct

# TODO-RADI: change to negation direct and indirect
def get_intervention_types():
    return [
        "negate_direct",
        "negate_indirect"
    ]


def load_examples(opts, train_opts):
    ''' Loads examples in pairs; returns a list of tuples, where each entry is the input for (non-negated version, negated version)
    '''
    # TODO: there is probably a more efficient way to load the dataset

    eval_collate_fn = nlvr2_triplet_eval_collate
    img_db = DetectFeatLmdb(opts.img_db,
                            train_opts.conf_th, train_opts.max_bb,
                            train_opts.min_bb, train_opts.num_bb,
                            opts.compressed_db)
    txt_db = TxtTokLmdb(opts.txt_db, -1)
    dset = Nlvr2TripletEvalDataset(txt_db, img_db, train_opts.use_img_type)

    data_dict = {}

    eval_dataloader = DataLoader(dset, batch_sampler=None,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=eval_collate_fn)

    for i, item in enumerate(eval_dataloader):
        data_dict[item['qids'][0]] = item

    corresponding_examples = []
    for id in dset.ids:
        if "-n-" in id:
            corresponding_id = "-".join(id.split("-", 4)[0:4])
            corresponding_tuple = (data_dict[corresponding_id], data_dict[id])
            corresponding_examples.append(corresponding_tuple)

    data_dict.clear()

    return corresponding_examples

def run_all(
    args,
    opts,
    model_type="gpt2",
    device="cuda",
    out_dir=".",
    random_weights=False,
    template_indices=None,
):
    print("Model:", model_type, flush=True)
    # Set up all the potential combinations.
    # TODO-RADI: we don't need these since we iterate over already constructed examples
    # professions = get_profession_list()
    # templates = get_template_list(template_indices)
    # TODO-RADI: implement the loading function
    train_opts = Struct(json.load(open(f'{args.train_dir}/log/hps.json')))
    loaded_examples = load_examples(args, train_opts)
    print(len(loaded_examples))

    intervention_types = get_intervention_types()
    print(intervention_types)
    # Initialize Model and Tokenizer.
    # TODO-RADI: initialise UNITER
    # TODO-RADI: we don't need the tokenizer cause the data is already preprocessed
    # tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    # TODO-RADI: model files; pass as argument
    ckpt_file = f'{opts.train_dir}/ckpt/model_step_{opts.ckpt}.pt'
    model_config = UniterConfig.from_json_file(
        f'{opts.train_dir}/log/model.json')

    # TODO-RADI: make sure to pass all necessary parameters
    model = Model(ckpt_file, model_config, device=device, gpt2_version=model_type, random_weights=random_weights)

    # Set up folder if it does not exist.
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string + "_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Iterate over all possible templates.
    # TODO-RADI: iterate over the constructed examples
    # for temp in templates:
    for example_pair in loaded_examples:
        # TODO-RADI: don't need this
        # print("Running template '{}' now...".format(temp), flush=True)
        # # Fill in all professions into current template
        # interventions = construct_interventions(temp, professions, tokenizer, device)
        # Consider all the intervention types
        for itype in intervention_types:
            print("\t Running with intervention: {}".format(itype), flush=True)
            # Run actual exp.
            # TODO-RADI: make sure the neuron_intervention_experiment function takes an example pair as input
            intervention_results = model.neuron_intervention_experiment(
                example_pair, itype, alpha=1.0
            )

            df = convert_results_to_pd(interventions, intervention_results)
            # Generate file name.
            temp_string = "_".join(temp.replace("{}", "X").split())
            model_type_string = model_type
            fname = "_".join([temp_string, itype, model_type_string])
            # Finally, save each exp separately.
            df.to_csv(os.path.join(base_path, fname + ".csv"))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hvd.init()
    parser = argparse.ArgumentParser(description="Run a set of neuron experiments.")
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


    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="""Model type [distilgpt2, gpt-2, etc.].""",
    )

    parser.add_argument(
        "--out_dir", default=".", type=str, help="""Path of the result folder."""
    )

    parser.add_argument(
        "--template_indices",
        nargs="+",
        type=int,
        help="Give the indices of templates if you want to run on only a subset",
    )

    parser.add_argument(
        "--randomize", default=False, action="store_true", help="Randomize model weights."
    )
    opts = parser.parse_args()


    # TODO-RADI: add the necessary arguments to load the databases
    run_all(
        opts,
        opts.model,
        device,
        opts.out_dir,
        random_weights=opts.randomize,
        template_indices=opts.template_indices,
    )
