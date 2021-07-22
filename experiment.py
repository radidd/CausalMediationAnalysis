
import math
import statistics
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
#from transformers import GPT2LMHeadModel, GPT2Tokenizer

#from attention_intervention_model import AttentionOverride
from utils_cma import batch, convert_results_to_pd

from UNITER.model.nlvr2 import UniterForNlvr2Triplet

from UNITER.utils.const import IMG_DIM

from apex import amp

np.random.seed(1)
torch.manual_seed(1)


class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self,
                 ckpt_file,
                 model_config,
                 opts,
                 device='cuda',
                 output_attentions=False,
                 random_weights=False,
                 gpt2_version='gpt2'):
        super()
        self.device = device

        # TODO-RADI: load UNITER
        # self.model = GPT2LMHeadModel.from_pretrained(
        #     gpt2_version,
        #     output_attentions=output_attentions)


        checkpoint = torch.load(ckpt_file)
        self.model = UniterForNlvr2Triplet(model_config, img_dim=IMG_DIM)
        self.model.init_type_embedding()
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        #print(type(self.model))
        #print(self.model)
        self.model, _ = amp.initialize(self.model, enabled=opts.fp16, opt_level='O2')
        self.model.eval()
        #print(type(self.model[0]))
        #print(self.model)
        # TODO-RADI: this might not work currently
        if random_weights:
            print('Randomizing weights')
            self.model.init_weights()

        #print(dir(self.model.uniter.embeddings))
        #print(dir(self.model.uniter.embeddings.word_embeddings))
        #print(self.model[0].uniter)
        #print(dir(self.model[0].uniter))
        #print(dir(self.model[0].uniter.encoder))
        #print(dir(self.model[0].uniter.encoder.layer[0]))
        # TODO-RADI: change to attributes of UNITER
        # Options
        self.top_k = 5
        # 12 for GPT-2
        #self.num_layers = len(self.model.transformer.h)
        self.num_layers = model_config.num_hidden_layers
        # 768 for GPT-2
        #self.num_neurons = self.model.transformer.wte.weight.shape[1]
        self.num_neurons = model_config.hidden_size
        # 12 for GPT-2
        #self.num_heads = self.model.transformer.h[0].attn.n_head
        self.num_heads = model_config.num_attention_heads


    def get_representations(self, context, position=0):
        # Hook for saving the representation
        # TODO-RADI: check how to make this save the [CLS] representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
            representations[layer] = output[0][position]
        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            # TODO-RADI: change to attributes of UNITER - find what the embedding layer is called
            handles.append(self.model.uniter.embeddings.word_embeddings.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=-1)))
            # hidden layers
            # TODO-RADI: change to attributes of UNITER - find what the hidden layers are called
            # TODO-RADI: BertLayer has attention, intermediate and output - which one should we intervene on?
            for layer in range(self.num_layers):
                handles.append(self.model.uniter.encoder.layer[layer]\
                                   .output.dropout.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=layer)))
            #print(context)
            # print(handles[0])
            # print(next(self.model.parameters()).device)
            # print(self.device)
            logits = self.model(context, compute_loss=False)
            #print(logits)
            for h in handles:
                h.remove()
        # print(representation[0][:5])
        return representation

    def get_probabilities_for_examples(self, context):
        """Return probabilities of single-token candidates given context"""
        # TODO-RADI: this can be simpler since there are only two options; make sure to get the probabilities from the linear classifier

        logits = self.model(context, compute_loss = False)
        probs = F.softmax(logits, dim=-1)
        return probs.tolist()

    def get_probabilities_for_examples_multitoken(self, context, candidates):
        """
        Return probability of multi-token candidates given context.
        Prob of each candidate is normalized by number of tokens.

        Args:
            context: Tensor of token ids in context
            candidates: list of list of token ids in each candidate

        Returns: list containing probability for each candidate
        """
        # TODO: Combine into single batch
        mean_probs = []
        context = context.tolist()
        for candidate in candidates:
            combined = context + candidate
            # Exclude last token position when predicting next token
            batch = torch.tensor(combined[:-1]).unsqueeze(dim=0).to(self.device)
            # Shape (batch_size, seq_len, vocab_size)
            logits = self.model(batch)[0]
            # Shape (seq_len, vocab_size)
            log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
            context_end_pos = len(context) - 1
            continuation_end_pos = context_end_pos + len(candidate)
            token_log_probs = []
            # TODO: Vectorize this
            # Up to but not including last token position
            for i in range(context_end_pos, continuation_end_pos):
                next_token_id = combined[i+1]
                next_token_log_prob = log_probs[i][next_token_id].item()
                token_log_probs.append(next_token_log_prob)
            mean_token_log_prob = statistics.mean(token_log_probs)
            mean_token_prob = math.exp(mean_token_log_prob)
            mean_probs.append(mean_token_prob)
        return mean_probs

    # TODO-RADI: we only have replace intervention type; change default;
    def neuron_intervention(self,
                            context,
                            outputs,
                            rep,
                            layers,
                            neurons,
                            position,
                            intervention_type='replace',
                            alpha=1.):
        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output,
                              position,
                              neurons,
                              intervention,
                              intervention_type):
            # Get the neurons to intervene on
            neurons = torch.LongTensor(neurons).to(self.device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            # print(input[0].shape)
            # print(output.shape)
            # print(neurons.shape)
            base = output[:, position, :].gather(
                1, neurons)
            intervention_view = intervention.view_as(base)

            if intervention_type == 'replace':
                base = intervention_view
            elif intervention_type == 'diff':
                base += intervention_view
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")
            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(output).byte()
            for i, v in enumerate(neurons):
                scatter_mask[i, position, v] = 1
            # Then take values from base and scatter
            output.masked_scatter_(scatter_mask, base.flatten())

        # Set up the context as batch
        batch_size = len(neurons)

        # TODO-RADI: can't batch this since the context is in dict form
        #context = context.unsqueeze(0).repeat(batch_size, 1)
        #print(context)

        batch_context = {}
        for key, value in context.items():
            try:
                if len(value.shape) == 2:
                    batch_context[key] = value.repeat(batch_size,1)

                elif len(value.shape) == 3:
                    batch_context[key] = value.repeat(batch_size, 1, 1)

            except AttributeError:
                continue

        handle_list = []
        for layer in set(layers):
          neuron_loc = np.where(np.array(layers) == layer)[0]
          n_list = []
          for n in neurons:
            unsorted_n_list = [n[i] for i in neuron_loc]
            n_list.append(list(np.sort(unsorted_n_list)))
          #print(n_list)
          #print(rep[layer].shape)
          intervention_rep = alpha * rep[layer][n_list]
          if layer == -1:
              wte_intervention_handle = self.model.uniter.embeddings.word_embeddings.register_forward_hook(
                  partial(intervention_hook,
                          position=position,
                          neurons=n_list,
                          intervention=intervention_rep,
                          intervention_type=intervention_type))
              handle_list.append(wte_intervention_handle)
          else:
              mlp_intervention_handle = self.model.uniter.encoder.layer[layer]\
                                            .output.dropout.register_forward_hook(
                  partial(intervention_hook,
                          position=position,
                          neurons=n_list,
                          intervention=intervention_rep,
                          intervention_type=intervention_type))
              handle_list.append(mlp_intervention_handle)
        new_probabilities = self.get_probabilities_for_examples(
            batch_context)
        for hndle in handle_list:
          hndle.remove()
        return new_probabilities


    # def neuron_intervention_experiment(self,
    #                                    word2intervention,
    #                                    intervention_type,
    #                                    layers_to_adj=[],
    #                                    neurons_to_adj=[],
    #                                    alpha=1,
    #                                    intervention_loc='all'):
    #     """
    #     run multiple intervention experiments
    #     """
    #     # TODO-RADI: probably don't need this function since we only have one intervention each time
    #     word2intervention_results = {}
    #     for word in tqdm(word2intervention, desc='words'):
    #         word2intervention_results[word] = self.neuron_intervention_single_experiment(
    #             word2intervention[word], intervention_type, layers_to_adj, neurons_to_adj,
    #             alpha, intervention_loc=intervention_loc)
    #
    #     return word2intervention_results

    # TODO-RADI: change bsize back to 800 when the parallel stuff is ok
    def neuron_intervention_single_experiment(self,
                                              intervention,
                                              intervention_type, layers_to_adj=[],
                                              neurons_to_adj=[],
                                              alpha=100,
                                              bsize=500, intervention_loc='all'):
        """
        run one full neuron intervention experiment
        """

        with torch.no_grad():
            '''
            Compute representations for base terms (one for each side of bias)
            '''

            # TODO-RADI: they compute everything three times for neutral, man, woman; we need it only twice for non-negated and negated;
            # TODO-RADI: check that the order of non-negated, negated is right
            # TODO-RADI: check what the position should be; position of the token whose representation we want?
            # TODO-RADI: default position to 0 which should be [CLS] but double check!!!
            orig_representations = self.get_representations(
                intervention[0],
                position=0)
            negated_representations = self.get_representations(
                intervention[1],
                position=0)

            #print(orig_representations)
            # TODO: this whole logic can probably be improved
            # determine effect type and set representations

            # TODO-RADI: change to our interventions; neg_direct and neg_indirect;
            # e.g. There are two dogs.
            if intervention_type == 'negate_direct':
                context = intervention[1]
                rep = orig_representations
                replace_or_diff = 'replace'
            # e.g. There aren't two dogs.
            elif intervention_type == 'negate_indirect':
                context = intervention[0]
                rep = negated_representations
                replace_or_diff = 'replace'
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")

            # Probabilities without intervention (Base case)
            # TODO-RADI: calculate for non-negated and negated; simplify cause candidate 1 and 2 are just true and false (only two candidates in prediction)
            candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_examples(
                intervention[0])[0]
            candidate1_alt1_prob, candidate2_alt1_prob = self.get_probabilities_for_examples(
                intervention[1])[0]


            # candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_examples(
            #     intervention.base_strings_tok[0].unsqueeze(0),
            #     intervention.candidates_tok)[0]
            # candidate1_alt1_prob, candidate2_alt1_prob = self.get_probabilities_for_examples(
            #     intervention.base_strings_tok[1].unsqueeze(0),
            #     intervention.candidates_tok)[0]
            # candidate1_alt2_prob, candidate2_alt2_prob = self.get_probabilities_for_examples(
            #     intervention.base_strings_tok[2].unsqueeze(0),
            #     intervention.candidates_tok)[0]

            # Now intervening on potentially biased example
            if intervention_loc == 'all':
              # TODO-RADI: might need to change this for UNITER; also can simplify since the probs add up to 1
              # TODO-RADI: explanation - i think this stores the output probabilities for the intervention on each of the neurons
              candidate1_probs = torch.zeros((self.num_layers + 1, self.num_neurons))
              candidate2_probs = torch.zeros((self.num_layers + 1, self.num_neurons))

              # TODO-RADI: check if this is correct for UNITER
              for layer in range(-1, self.num_layers):
                for neurons in batch(range(self.num_neurons), bsize):
                    neurons_to_search = [[i] + neurons_to_adj for i in neurons]
                    layers_to_search = [layer] + layers_to_adj

                    probs = self.neuron_intervention(
                        context=context,
                        outputs=[0,1],
                        rep=rep,
                        layers=layers_to_search,
                        neurons=neurons_to_search,
                        position=0,
                        intervention_type=replace_or_diff,
                        alpha=alpha)
                    # probs = self.neuron_intervention(
                    #     context=context,
                    #     outputs=intervention.candidates_tok,
                    #     rep=rep,
                    #     layers=layers_to_search,
                    #     neurons=neurons_to_search,
                    #     position=intervention.position,
                    #     intervention_type=replace_or_diff,
                    #     alpha=alpha)

                    for neuron, (p1, p2) in zip(neurons, probs):
                        candidate1_probs[layer + 1][neuron] = p1
                        candidate2_probs[layer + 1][neuron] = p2
                        # Now intervening on potentially biased example
            elif intervention_loc == 'layer':
              layers_to_search = (len(neurons_to_adj) + 1)*[layers_to_adj]
              candidate1_probs = torch.zeros((1, self.num_neurons))
              candidate2_probs = torch.zeros((1, self.num_neurons))

              for neurons in batch(range(self.num_neurons), bsize):
                neurons_to_search = [[i] + neurons_to_adj for i in neurons]

                probs = self.neuron_intervention(
                    context=context,
                    outputs=intervention.candidates_tok,
                    rep=rep,
                    layers=layers_to_search,
                    neurons=neurons_to_search,
                    position=intervention.position,
                    intervention_type=replace_or_diff,
                    alpha=alpha)
                for neuron, (p1, p2) in zip(neurons, probs):
                    candidate1_probs[0][neuron] = p1
                    candidate2_probs[0][neuron] = p2
            else:
              probs = self.neuron_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        rep=rep,
                        layers=layers_to_adj,
                        neurons=neurons_to_adj,
                        position=intervention.position,
                        intervention_type=replace_or_diff,
                        alpha=alpha)
              for neuron, (p1, p2) in zip(neurons_to_adj, probs):
                  candidate1_probs = p1
                  candidate2_probs = p2


        return (candidate1_base_prob, candidate2_base_prob,
                candidate1_alt1_prob, candidate2_alt1_prob,
                candidate1_probs, candidate2_probs)


def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model(device=DEVICE)

    base_sentence = "The {} said that"
    biased_word = "teacher"
    intervention = Intervention(
            tokenizer,
            base_sentence,
            [biased_word, "man", "woman"],
            ["he", "she"],
            device=DEVICE)
    interventions = {biased_word: intervention}

    intervention_results = model.neuron_intervention_experiment(
        interventions, 'man_minus_woman')
    df = convert_results_to_pd(
        interventions, intervention_results)
    print('more probable candidate per layer, across all neurons in the layer')
    print(df[0:5])


if __name__ == "__main__":
    main()
