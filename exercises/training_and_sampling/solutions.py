# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.chdir("..")
from transformer_from_scratch.solutions import lm_cross_entropy_loss, Config, DemoTransformer
os.chdir("training_and_sampling")
# print(os.getcwd())

# from IPython import get_ipython
# ipython = get_ipython()
# ipython.magic("load_ext autoreload")
# ipython.magic("autoreload 2")

MAIN = __name__ == "__main__"

import re
import torch as t
from torch.utils.data import Dataset, DataLoader
import transformers
from typing import List, Tuple, Union, Optional, Callable
from torch import nn, optim
import numpy as np
from einops import einsum, rearrange, repeat, reduce
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import torch.nn.functional as F
import datasets
from torchtyping import TensorType as TT

device = t.device("cuda" if t.cuda.is_available() else "cpu")

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

# %%


# Load the text data
if MAIN:
    with open("shakespeare-corpus.txt", encoding="utf-8") as file:
        text = file.read()

        while "  " in text:
            text = re.sub("  ", " ", text)
        while "\n\n\n" in text:
            text = re.sub("\n\n\n", "\n\n", text)

        # text = re.split(r"\b", text)

        # tokens = tokenizer.encode(text, return_tensors="pt")
        # words = re.split(r"\b", text)

# %%

class SimpleTokenizer():

    def __init__(self, text: str):
        self.text = text
        self.words = sorted(set(re.split(r"\b", text)))
        self.unk = len(self.words) + 1
        self.bos_token_id = len(self.words) + 2
        self.word_to_index = {word: index for index, word in enumerate(self.words)}
        self.index_to_word = {index: word for index, word in enumerate(self.words)}

    def encode(self, input_text, return_tensors: Optional[str] = None) -> Union[List, t.Tensor]:
        '''
        Tokenizes and encodes the input text.

        If `return_tensors` is None, should return list of Python integers.
        If `return_tensors` is "pt", should return a PyTorch tensor of shape (1, num_tokens).
        '''
        split_text = re.split(r"\b", input_text)
        encoding = [self.word_to_index.get(word, self.unk) for word in split_text]
        if self.unk in encoding:
            print(f"Warning: Unknown token found in input text")
        if return_tensors == "pt":
            return t.tensor(encoding).unsqueeze(0)
        return encoding

    def decode(self, tokens: t.Tensor):
        '''
        Decodes the tokens into a string of text.
        '''
        return "".join([self.index_to_word[token] for token in tokens])


if MAIN:
    mytokenizer = SimpleTokenizer(text)

# %%

def test_tensor_dataset(TensorDataset):
    tensors = [t.rand((10, 20)), t.rand((10, 5)), t.arange(10)]
    dataset = TensorDataset(*tensors)
    assert len(dataset) == 10
    for index in [0, slice(0, 5, 1), slice(1, 5, 2)]:
        print("Testing with index:", index)
        expected = tuple(tensor[index] for tensor in tensors)
        actual = dataset[index]
        for e, a in zip(expected, actual):
            t.testing.assert_close(e, a)
    print("All tests in `test_tensor_dataset` passed!")

class TensorDataset:
    def __init__(self, *tensors: t.Tensor):
        """Validate the sizes and store the tensors in a field named `tensors`."""
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        assert len(set(batch_sizes)) == 1, "All tensors must have the same size in the first dimension"
        self.tensors = tensors

    def __getitem__(self, index: Union[int, slice]) -> Tuple[t.Tensor, ...]:
        """Return a tuple of length len(self.tensors) with the index applied to each."""
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        """Return the size in the first dimension, common to all the tensors."""
        return self.tensors[0].shape[0]


if MAIN:
    test_tensor_dataset(TensorDataset)

# %%

@dataclass
class TransformerTrainingArgs():
    tokenizer: transformers.PreTrainedTokenizer = mytokenizer
    epochs: int = 3
    batch_size: int = 16
    max_seq_len: int = 48
    optimizer: Callable[..., t.optim.Optimizer] = t.optim.Adam
    optimizer_args: Tuple = ()
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    filename_save_model: str = "transformer_shakespeare.pt"


# %%

def tokenize_text(text: str, max_seq_len: int, tokenizer: SimpleTokenizer):
    '''
    Takes a string of text, and returns an array of tokens rearranged into chunks of size max_seq_len.
    '''
    tokens: TT[1, "num_tokens"] = tokenizer.encode(text, return_tensors="pt")

    # We want to rearrange the tokens into chunks of size max_seq_len.
    num_tokens = tokens.size(1) - (tokens.size(1) % max_seq_len)
    tokens = rearrange(
        tokens[0, :num_tokens], "(chunk seq_len) -> chunk seq_len", seq_len=max_seq_len
    )

    # Append the start token to the beginning of each chunk
    tokens = t.cat([
        t.full((tokens.size(0), 1), tokenizer.bos_token_id, dtype=t.long), 
        tokens
    ], dim=1)

    return tokens


if MAIN:
    tokens = tokenize_text(text[:500], max_seq_len=48, tokenizer=mytokenizer)
    print("Does this size look reasonable for the first 500 characters?\n", tokens.shape)

# %%

def train_transformer(model: DemoTransformer, text: str, args: TransformerTrainingArgs) -> Tuple[list, list]:
    '''
    Trains an autoregressive transformer on the data in the trainset.

    Returns tuple of (train_loss, test_loss), containing the cross entropy losses for the thing.
    '''
    model.to(args.device)

    tokens = tokenize_text(text, max_seq_len=args.max_seq_len, tokenizer=args.tokenizer)

    len_trainset = int(0.9 * len(tokens))
    trainset = TensorDataset(tokens[:len_trainset])
    testset = TensorDataset(tokens[len_trainset:])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    optimizer = args.optimizer(model.parameters(), *args.optimizer_args)
    train_loss_list = []
    test_loss_list = []

    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)
        for (tokens,) in progress_bar:

            tokens = tokens.to(args.device)

            logits = model(tokens)
            loss = lm_cross_entropy_loss(logits, tokens)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss_list.append(loss.item())

            progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss = {loss:.3f}")

        with t.inference_mode():

            test_loss = 0.0
            total = 0

            progress_bar = tqdm(testloader, desc="Calculating test loss")
            for (tokens,) in progress_bar:

                tokens = tokens.to(args.device)

                logits = model(tokens)

                test_loss += lm_cross_entropy_loss(logits, tokens) * tokens.size(0)
                total += tokens.size(0)

            test_loss /= total
            test_loss_list.append(test_loss.item())

        print(f"Train loss = {loss:.4f}, Test loss = {test_loss:.4f}")

    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return train_loss_list, test_loss_list

# %%

if MAIN:
    config = Config(
        d_model = 384,
        layer_norm_eps = 1e-5,
        d_vocab = 50257,
        init_range = 0.02,
        n_ctx = 1024,
        d_head = 64,
        d_mlp = 1536,
        n_heads = 12,
        n_layers = 6
    )

    model = DemoTransformer(config)

    args = TransformerTrainingArgs(
        tokenizer = mytokenizer,
        batch_size = 64,
        epochs = 3,
    )

    train_loss_list, test_loss_list = train_transformer(model, text, args)

    fig = go.Figure(
        data = [
            go.Scatter(y=train_loss_list, x=np.arange(len(train_loss_list)) * args.batch_size, name="Train"),
            go.Scatter(y=test_loss_list, x=np.arange(len(test_loss_list)) * 33579, name="Test"),
        ],
        layout = go.Layout(
            title = "Training loss for autoregressive transformer, on Shakespeare corpus",
            xaxis_title = "Number of sequences seen",
            yaxis_title = "Cross entropy loss",
        )
    )
    fig.show()

# %%





def greedy_search(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    """
    out = logits.argmax().item()
    assert isinstance(out, int)
    return out

def sample_basic(logits: t.Tensor) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    """
    distribution = t.distributions.categorical.Categorical(logits=logits)
    out = distribution.sample().item()
    assert isinstance(out, int)
    return out

def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    """
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    """
    assert temperature > 0
    return logits / temperature

def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    """
    input_ids: shape (seq, )
    logits: shape (vocab_size, )
    Return: shape (vocab_size, )
    """
    (vocab_size,) = logits.shape
    id_freqs = t.bincount(input_ids, minlength=vocab_size)
    return logits - freq_penalty * id_freqs

def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    """
    top_logits, top_idx = t.topk(logits, top_k)
    idx = t.distributions.categorical.Categorical(logits=top_logits).sample()
    return top_idx[idx].item()

def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    """
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    Return: a sampled token
    """
    logits_sorted, indices = logits.sort(descending=True, stable=True)
    cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
    n_keep = t.searchsorted(cumul_probs, top_p, side="right").item() + 1
    n_keep = max(n_keep, min_tokens_to_keep)
    keep_idx = indices[:n_keep]
    keep_logits = logits[keep_idx]
    sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
    return keep_idx[sample].item()

# %%

def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def sample_tokens(
    model: DemoTransformer,
    tokenizer: transformers.PreTrainedTokenizer,
    initial_text: str,
    max_tokens_generated=30,
    **kwargs # kwargs are for params like temperature, top_k, etc
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    # Note - an alternative to model.eval() is to use the @t.inference_mode() decorator for this whole function.
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.long, device=device)
        new_input_ids_window = new_input_ids[-min(args.max_seq_len, new_input_ids.shape[0]):].unsqueeze(0)
        logits = model(new_input_ids_window)[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

# %%

if MAIN:
    initial_text = "Turn down for what"

    text_output = sample_tokens(model, mytokenizer, initial_text, max_tokens_generated=100, temperature=1.0, top_k=10)

    print(text_output)

# Result:

# turn down for what you do you think,
# That take the last, of many, which is so much I
# As this blows along than my life thou say’st, which makes thy hand,
# Thou wilt be given, or more
# Entitled in thy great world’s fresh blood will,
# To answer th’ alluring countenance, beauty 

# %%









# %%

@t.inference_mode()
def beam_search(
    model, input_ids: t.Tensor, num_return_sequences: int, num_beams: int, max_new_tokens: int, tokenizer, verbose=False
) -> List[Tuple[float, t.Tensor]]:
    """
    input_ids: (seq, ) - the prompt

    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    """
    assert num_return_sequences <= num_beams

    model.eval()
    
    # Create list to store the sequences to return
    # We only add to this when we generate an EOS token, or at the very end
    final_logitsums_and_completions = []

    # Create list to store the current best completions and their logit scores
    best_logitsums_and_completions = [(0, input_ids.tolist())]

    for n in tqdm(range(max_new_tokens)):
        
        # Create a list to store the completions at this stage
        new_best_logitsums_and_completions = []

        # This section loops through all completions so far, and get the next words
        for (logitsum, completion) in best_logitsums_and_completions:

            # Get output (we only care about the vector of logits for the next token)
            output = model(t.tensor(completion).unsqueeze(0).to(device, t.long))
            output = (output if isinstance(output, t.Tensor) else output.logits)[0, -1, :].log_softmax(-1)

            # Find the top `num_beams` (because this is the maximum we might need)
            topk_logits, topk_indices = t.topk(output, k=num_beams)

            # Append to the new best completions list
            for logit, idx in zip(topk_logits, topk_indices):
                new_completion_and_logit = (logitsum + logit.item(), completion + [idx.item(),])
                new_best_logitsums_and_completions.append(new_completion_and_logit)

        # This section updates (and sorts) the list of best completions, and also updates `final_logitsums_and_completions` if EOS was produced
        best_logitsums_and_completions = []
        for (logitsum, completion) in sorted(new_best_logitsums_and_completions, key=lambda x: x[0], reverse=True):
            
            # If token is eos then add it to final_logitsums_and_completions
            if completion[-1] == getattr(tokenizer, "eos_token_id", None):
                final_logitsums_and_completions.append((logitsum, completion))
            
            # Else add it to best_logitsums_and_completions until that list is full up, then we break out of for loop
            else:
                best_logitsums_and_completions.append((logitsum, completion))
                if len(best_logitsums_and_completions) == num_beams:
                    break

        # Add `best_logitsums_and_completions` to our final list, if necessary
        # Also sort the final completions list, and print output if necessary
        if n == max_new_tokens - 1:
            final_logitsums_and_completions.extend(best_logitsums_and_completions)
            final_logitsums_and_completions = sort_by_logits_and_crop(final_logitsums_and_completions, max_size=num_return_sequences)
            if verbose: print_sequences(f"Returning best {num_return_sequences=} completions:", final_logitsums_and_completions, tokenizer)
        else:
            final_logitsums_and_completions = sort_by_logits_and_crop(final_logitsums_and_completions, max_size=num_beams)
            if verbose: print_sequences(f"Printing {num_beams=} best completions:", best_logitsums_and_completions, tokenizer)

    return final_logitsums_and_completions

def print_sequences(name, logitsums_and_completions, tokenizer):
    if len(logitsums_and_completions) == 0:
        return
    print("\n" + name + "\n")
    print("logitsum | completion")
    for logit_sum, completion in logitsums_and_completions:
        text = tokenizer.decode(completion)
        print(f"{logit_sum:>8.3f} | {text}")

def sort_by_logits_and_crop(logitsums_and_completions, max_size):
    logitsums_and_completions = sorted(logitsums_and_completions, key=lambda x: x[0], reverse=True)
    logitsums_and_completions = logitsums_and_completions[:min(max_size, len(logitsums_and_completions))]
    return logitsums_and_completions



# %%

if MAIN:
    initial_text = "My lord"
    input_ids = mytokenizer.encode(initial_text, return_tensors="pt").squeeze()
    num_return_sequences = 5
    num_beams = 10
    max_new_tokens = 20

    final_logitsums_and_completions = beam_search(model, input_ids, num_return_sequences, num_beams, max_new_tokens, mytokenizer, verbose=True)

    text = mytokenizer.decode(final_logitsums_and_completions[0][1])
    print(text)

# %%

# Note - this beam search isn't very good on this model, because it's not very well trained. So you can try and load the full model to see how it's meant to look:

# Or, all of this sampling stuff can be the loaded-in GPT