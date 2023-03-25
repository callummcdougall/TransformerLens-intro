import os
# if not os.path.exists("./images"):
#     os.chdir("./ch6")
import re, json
import plotly.io as pio

from st_dependencies import *
styling()

def img_to_html(img_path, width):
    with open("images/page_images/" + img_path, "rb") as file:
        img_bytes = file.read()
    encoded = base64.b64encode(img_bytes).decode()
    return f"<img style='width:{width}px;max-width:100%;st-bottom:25px' src='data:image/png;base64,{encoded}' class='img-fluid'>"
def st_image(name, width):
    st.markdown(img_to_html(name, width=width), unsafe_allow_html=True)

def read_from_html(filename):
    filename = f"images/{filename}.html" if "written_images" in filename else f"images/page_images/{filename}.html"
    with open(filename) as f:
        html = f.read()
    call_arg_str = re.findall(r'Plotly\.newPlot\((.*)\)', html)[0]
    call_args = json.loads(f'[{call_arg_str}]')
    try:
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    except:
        del call_args[2]["template"]["data"]["scatter"][0]["fillpattern"]
        plotly_json = {'data': call_args[1], 'layout': call_args[2]}
        fig = pio.from_json(json.dumps(plotly_json))
    return fig

NAMES = []

def complete_fig_dict(fig_dict):
    for name in NAMES:
        if name not in fig_dict:
            fig_dict[name] = read_from_html(name)
    return fig_dict
if "fig_dict" not in st.session_state:
    st.session_state["fig_dict"] = {}
fig_dict_old = st.session_state["fig_dict"]
fig_dict = complete_fig_dict(fig_dict_old)
if len(fig_dict) > len(fig_dict_old):
    st.session_state["fig_dict"] = fig_dict

def section_home():
    st.sidebar.markdown(r"""
## Table of contents

<ul class="contents">
    <li><a class="contents-el" href="#introduction">Introduction</a></li>
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#learning-objectives">Learning Objectives</a></li>
</ul>""", unsafe_allow_html=True)
#     st.markdown(r"""
# Links to Colab: [**exercises**](https://colab.research.google.com/drive/1LpDxWwL2Fx0xq3lLgDQvHKM5tnqRFeRM?usp=share_link), [**solutions**](https://colab.research.google.com/drive/1ND38oNmvI702tu32M74G26v-mO5lkByM?usp=share_link)
# """)
    st_image("sampling.png", 350)
    st.markdown(r"""
# Training and Sampling

Coming soon!
""")
    st.markdown(r"""
## Learning objectives

Here are the learning objectives for each section of the tutorial. At the end of each section, you should refer back here to check that you've understood everything.
""")

    st.info(r"""
## 1️⃣ Training

* Review the interpretation of a transformer's output, and learn how it's trained by minimizing cross-entropy loss between predicted and actual next tokens
* Construct datasets and dataloaders for the corpus of Shakespeare text
* Implement a transformer training loop
""")
    st.info(r"""
## 2️⃣ Sampling and Caching

* Learn how to sample from a transformer
    * This includes basic methods like greedy search or top-k, and more advanced methods like beam search
* Learn how to cache the output of a transformer, so that it can be used to generate text more efficiently
""")

    
def section_training():
    st.sidebar.markdown(r"""
## Table of contents

<ul class="contents">
    <li><a class="contents-el" href="#imports">Imports</a></li>
    <li><a class="contents-el" href="#cross-entropy-loss">Cross entropy loss</a></li>
    <li><a class="contents-el" href="#tokenizers">Tokenizers</a></li>
    <li><a class="contents-el" href="#preparing-text">Preparing text</a></li>
    <li><a class="contents-el" href="#datasets-and-dataloaders">Datasets and Dataloaders</a></li>
    <li><a class="contents-el" href="#training-loop">Training loop</a></li>

</ul>""", unsafe_allow_html=True)
    st.markdown(r"""
# Training
""")
    st.info(r"""
### Learning Objectives

* Review the interpretation of a transformer's output, and learn how it's trained by minimizing cross-entropy loss between predicted and actual next tokens
* Construct datasets and dataloaders for the corpus of Shakespeare text
* Implement a transformer training loop
""")
    st.markdown(r"""
Hopefully, you've now successfully implemented a transformer, and seen how to use it to generate output autoregressively. You might also have seen the example training loop at the end of the last section. Here, you'll train your transformer in a more hands-on way, using the [complete works of William Shakespeare](https://www.gutenberg.org/files/100/100-0.txt).

This is the task recommended by Jacob Hilton in his [curriculum](https://github.com/jacobhilton/deep_learning_curriculum).

## Imports

```python
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.chdir("..")
from transformer_from_scratch.solutions import lm_cross_entropy_loss, Config, DemoTransformer
os.chdir("training_and_sampling")

MAIN = __name__ == "__main__"

import re
import torch as t
from torch.utils.data import DataLoader
import transformers
from typing import List, Tuple, Union, Optional, Callable, Dict
import numpy as np
import einops
from dataclasses import dataclass
import plotly.graph_objects as go
from tqdm import tqdm
from torchtyping import TensorType as TT

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

## Cross entropy loss

Your transformer's input has shape `(batch, seq_len)`, where the `[i, j]`-th element is the token id of the `j`-th token in the `i`-th sequence. Your transformer's output has shape `(batch, seq_len, vocab_size)`, where the `[i, j, :]`-th element is a vector of logits, representing a probability distribution over the token that **follows** the `j`-th token in the `i`-th sequence.

When training our model, we use cross-entropy loss between the model's predictions and the actual next tokens. In other words, we can take the `[:, :-1, :]`-th slice of our output (which is a tensor of probability distributions for the **last** `seq_len - 1` tokens in each sequence), and compare this to the `[:, 1:, :]`-th slice (which represents the actual tokens we're trying to predict).

In the last section, we saw the function `lm_cross_entropy_loss` which calculated this for us. Let's take another look at this function, so we understand how it works:

```python
def lm_cross_entropy_loss(logits: t.Tensor, tokens: t.Tensor):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()
```

First, we get `log_probs`, which are the log probabilities of each token in the vocab. Log probs are (as you'd probably guess!) the log of the probabilities implied by the logit distribution. We get them from logits by taking softmax, then taking log again (so they're equal to logits, up to a constant difference). If you examine the formula for [cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), you'll notice that it's just the negative of the log probability of the correct token.

In the second line, we use the `gather` method to take the log probabilities corresponding to the correct token. This is a bit confusing, and you don't need to understand the exact syntax of `gather`. This line of code does the following:
* Indexes `log_probs`, taking the `[:, :-1]`-th slice (so we have the logits corresponding to the **last** `seq_len - 1` tokens in each sequence)
* Indexes `tokens`, taking the `[:, 1:]`-th slice (so we have the actual tokens we're trying to predict)
* Indexes into the reduced `log_probs` tensor using `gather`, so we get the log probabilities of the correct tokens

Finally, we take the mean of the negative log probabilities, and return this as our loss. Remember that log probs are always negative (because log of a number less than 1 is negative), so our loss will always be non-negative. It will tend to zero only if our model tends towards assigning 100% probability to the correct token, and 0% to all others.

## Tokenizers

Now that we've got cross entropy loss out of the way, let's start working with our dataset. We'll be using the Shakespeare corpus for this exercises; you can get the text as follows:

```python
with open("shakespeare-corpus.txt", encoding="utf-8") as file:
    text = file.read()
```

You should print out the first few lines of this text, and get a feel for what it looks like.

Rather than using a fancy tokenizer, we'll just split the text into tokens using a regular expression. This is a bit crude, but it's good enough for our purposes.
""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `SimpleTokenizer`

Below, you should fill in the `SimpleTokenizer` class. Some guidance for this exercise:

#### __init__

The `text` argument is meant to be a string (this will be the same as the `text` object you defined above). Here, you should define `self.words` as a list of all the different tokens that appear in the text, sorted in some reasonable way (you can split the text with `re.split(r"\b", text))`). You should then define `self.word_to_index` and `self.index_to_word`, which are dictionaries that map tokens to their token ids, and vice-versa (with the token ids being the positions of the tokens in `self.words`).

Also, it's good practice to include an unknown token `unk` in your vocabulary, just in case you feed the model a token that it hasn't seen before. We won't bother using a start token here (although you might want to think about doing this, as a bonus exercise).

#### `encode`

This takes in some text, and returns tokens. If `return_tensors` is None (the default), this should return a simple list of integers. If `return_tensors == "pt"`, this should return a PyTorch tensor of shape `(1, seq_len)` (it's good practice to always add a batch dimension, even if there's only one sequence in the batch).

If the input text contains an unknown token, then you can print an error message (or raise an exception).

#### `decode`

Finally, this should take in a list or tensor of tokens (you can assume that the batch dimension will be 1 if it's a tensor), and returns a string of the decoded text.
        
```python
class SimpleTokenizer():

    def __init__(self, text: str):
    pass

    def encode(self, input_text, return_tensors: Optional[str] = None) -> Union[List, t.Tensor]:
        '''
        Tokenizes and encodes the input text.

        If `return_tensors` is None, should return list of Python integers.
        If `return_tensors` is "pt", should return a PyTorch tensor of shape (1, num_tokens).
        '''
        pass

    def decode(self, tokens: Union[List, t.Tensor]):
        '''
        Decodes the tokens into a string of text.
        '''
        pass


if MAIN:
    mytokenizer = SimpleTokenizer(text)
""")
        with st.expander("Solution"):
            st.markdown(r"""
class SimpleTokenizer():

    def __init__(self, text: str):
        self.text = text
        self.words = sorted(set(re.split(r"\b", text)))
        self.unk = len(self.words) + 1
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

    def decode(self, tokens: Union[List, t.Tensor]):
        '''
        Decodes the tokens into a string of text.
        '''
        if isinstance(tokens, t.Tensor) and tokens.dim() == 2:
            assert tokens.size(0) == 1, "Only batch size 1 is supported"
            tokens = tokens[0]
        return "".join([self.index_to_word[token] for token in tokens])
""")
            
    st.markdown(r"""
## Preparing text

We have our tokenizer, but we still need to be able to take in our `text` object and turn it into a tensor of token ids, without any of them overlapping. This is important because overlapping sequences might cause use to double-count certain sequences during training, and will make it seem like our model is learning faster than it really is.


""")
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - implement `prepare_text`

Below, you should fill in the `prepare_text` function.

```python
def prepare_text(text: str, max_seq_len: int, tokenizer: SimpleTokenizer):
    '''
    Takes a string of text, and returns an array of tokens rearranged into chunks of size max_seq_len.
    '''
    pass


if MAIN:
    tokens = prepare_text(text[:500], max_seq_len=48, tokenizer=mytokenizer)
    print("Does this size look reasonable for the first 500 characters?\n", tokens.shape)
```
""")
        with st.expander("Hint"):
            st.markdown(r"""
This exercise just involves encoding the text, then rearranging the size `(1, num_tokens)` tensor into a 2D tensor of shape `(batch, max_seq_len)`. You'll have to crop some tokens off the end, if the number of tokens doesn't exactly divide by `max_seq_len`.
""")
        with st.expander("Solution"):
            st.markdown(r"""
def prepare_text(text: str, max_seq_len: int, tokenizer: SimpleTokenizer):
    '''
    Takes a string of text, and returns an array of tokens rearranged into chunks of size max_seq_len.
    '''
    tokens: TT[1, "num_tokens"] = tokenizer.encode(text, return_tensors="pt")

    # We want to rearrange the tokens into chunks of size max_seq_len.
    num_tokens = tokens.size(1) - (tokens.size(1) % max_seq_len)
    tokens = einops.rearrange(
        tokens[0, :num_tokens], "(chunk seq_len) -> chunk seq_len", seq_len=max_seq_len
    )

    return tokens
""")
            
    st.markdown(r"""
## Datasets and Dataloaders

Finally, we'll create dataset objects, which can be read by a `DataLoader`. We've just given you the code below, because the exercise isn't too different from things you've already done. Note that the dataset class takes in `*tensors`, i.e. a list of tensors, with each one having shape `(batch, ...)`. For instance, we might want to create a dataset with images and class labels, with each of those being a separate tensor with first dimension as batch size. However, here we'll only be passing in a single tensor; our token ids.

```python
class TensorDataset:
    def __init__(self, *tensors: t.Tensor):
        '''Validate the sizes and store the tensors in a field named `tensors`.'''
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        assert len(set(batch_sizes)) == 1, "All tensors must have the same size in the first dimension"
        self.tensors = tensors

    def __getitem__(self, index: Union[int, slice]) -> Tuple[t.Tensor, ...]:
        '''Return a tuple of length len(self.tensors) with the index applied to each.'''
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        '''Return the size in the first dimension, common to all the tensors.'''
        return self.tensors[0].shape[0]

        
if MAIN:
    dataset = TensorDataset(tokens)
```

You should play around with this dataset object, and make sure you understand how it works.
""")    
    st.markdown(r"""
## Training loop

Now, it's time for our training loop! We've left this exercise very open-ended, like our implementation of the ResNet training loop in last week's exercises. The principles are exactly the same, and we've provided you with a skeleton of the function to help get you started. Again, we use a `dataclass` object to store the training parameters, because this is a useful way of keeping your code organised.
""")
    
    with st.columns(1)[0]:
        st.markdown(r"""
#### Exercise - write a training loop
        
```python
@dataclass
class TransformerTrainingArgs():
    tokenizer: transformers.PreTrainedTokenizer = mytokenizer
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 48
    optimizer: Callable[..., t.optim.Optimizer] = t.optim.Adam
    optimizer_kwargs: Dict = dict(lr=0.001, betas=(0.9, 0.999))
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    filename_save_model: str = "transformer_shakespeare.pt"

    
def train_transformer(model: DemoTransformer, text: str, args: TransformerTrainingArgs) -> Tuple[list, list]:
    '''
    Trains an autoregressive transformer on the data in the trainset.

    Returns tuple of (train_loss, test_loss), containing the cross entropy losses for the thing.
    '''
    model.to(args.device)

    tokens = prepare_text(text, max_seq_len=args.max_seq_len, tokenizer=args.tokenizer)

    randperm = t.randperm(tokens.size(0))
    len_trainset = int(0.9 * tokens.size(0))
    trainset = TensorDataset(tokens[randperm[:len_trainset]])
    testset = TensorDataset(tokens[randperm[len_trainset:]])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    optimizer = args.optimizer(model.parameters(), **args.optimizer_kwargs)

    
    # YOUR CODE HERE - implement training and testing loops

    
    print(f"\nSaving model to: {args.filename_save_model}")
    t.save(model, args.filename_save_model)
    return train_loss_list, test_loss_list
    ```

You can take a look at the solutions for an example implementation (although it's totally fine to have something which looks different to this).

Once you've written a training loop, you can run it (and plot your output) with the following code:

```python
if MAIN:
    config = Config(
        d_model = 384,
        layer_norm_eps = 1e-5,
        d_vocab = 50257,
        init_range = 0.02,
        n_ctx = 1024,
        d_head = 64,
        d_mlp = 1536,
        n_heads = 6,
        n_layers = 4
    )

    model = DemoTransformer(config)

    args = TransformerTrainingArgs(
        tokenizer = mytokenizer,
        batch_size = 8,
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
```

You can try playing around with some of the hyperparameters, and see how they affect the training process. You might also want to try out using different datasets (there are many online you can use!).
""")
    
def section_sampling():
    st.sidebar.markdown(r"""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#sampling-boilerplate">Sampling Boilerplate</a></li>
    <li><a class="contents-el" href="#greedy-search">Greedy Search</a></li>
    <li><a class="contents-el" href="#sampling-with-categorical">Sampling with <code>Categorical</code></a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#temperature">Temperature</a></li>
        <li><a class="contents-el" href="#frequency-penalty">Frequency Penality</a></li>
        <li><a class="contents-el" href="#sampling-manual-testing">Sampling - Manual Testing</a></li>
    </ul></li>
    <li><a class="contents-el" href="#top-k-sampling">Top-K Sampling</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#top-k-sampling-example">Top-K Sampling - Example</a></li>
    </ul></li>
    <li><a class="contents-el" href="#top-p-aka-nucleus-sampling">Top-p aka Nucleus Sampling</a></li>
    <li><ul class="contents">
        <li><a class="contents-el" href="#top-p-sampling-example">Top-p Sampling - Example</a></li>
    </ul></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Sampling
""")
    st.info(r"""
#### Learning Objectives

* Learn how to sample from a transformer
    * This includes basic methods like greedy search or top-k, and more advanced methods like beam search
""")
    st.markdown(r"""
One obvious method to sample tokens from a distribution would be to always take the token assigned the highest probability. But this can lead to some boring and repetitive outcomes, and at worst it can lock our transformer's output into a loop.

First, you should read HuggingFace's blog post [How to generate text: using different decoding methods for language generation with Transformers
](https://huggingface.co/blog/how-to-generate). Once you've done that, we've included some exercises below that will allow you to write your own methods for sampling from a transformer. Some of the exercises are strongly recommended (two asterisks), some are weakly recommended (one asterisk) and others are perfectly fine to skip if you don't find these exercises as interesting.

We will be working with the [HuggingFace implementation](https://huggingface.co/docs/transformers/index) of classic transformer models like GPT. You might have to install the transformers library before running the cells below. You may also want to go back and complete exercise 2 in the [Tokenisation and Embedding exercises](https://arena-ldn-w1d1.streamlitapp.com/Tokenisation_and_embedding) from W1D1.

```python
import torch as t
import torch.nn.functional as F
import transformers

gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
```

## Sampling Boilerplate

The provided functions `apply_sampling_methods` and `sample_tokens` include the boilerplate for sampling from the model. Note that there is a special token `tokenizer.eos_token`, which during training was added to the end of a each article. GPT-2 will generate this token when it feels like the continuation is at a reasonable stopping point, which is our cue to stop generation.

The functions called in `apply_sampling_methods` are not defined yet - you are going to implement them below.

```python

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
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.int64, device=device)
        new_input_ids_truncated = new_input_ids[-min(tokenizer.model_max_length, new_input_ids.shape[0]):].unsqueeze(0)
        output = model(new_input_ids_truncated)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        assert isinstance(new_token, int)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)
```

A few notes on this function:

* We use `tokenizer.encode` to convert the initial text string into a list of logits. You can also pass the argument `return_tensors="pt"` in order to return the output as a tensor.
* `new_input_ids` is a concatenation of the original input ids, and the ones that have been autoregressively generated.
* `new_input_ids_truncated` truncates `new_input_ids` at `max_seq_len` (because you might get an error at the positional embedding stage if your input sequence length is too large).
* The line `all_logits = ...` is necessary because HuggingFace's GPT doesn't just output logits, it outputs an object which contains `logits` and `past_key_values`. In contrast, your model will probably just output logits, so we can directly define logits as the model's output.
""")
    with st.expander("Question - why do we take logits[0, -1] ?"):
        st.markdown("""
Our model input has shape `(batch, seq_len)`, and each element is a token id. Our output has dimension `(batch, seq_len, vocab_size)`, where the `[i, j, :]`th element is a vector of logits representing a prediction for the `j+1`th token.

In this case, our batch dimension is 1, and we want to predict the token that follows after all the tokens in the sequence, hence we want to take `logits[0, -1, :]`.
""")

    st.markdown("""
### Greedy Search

Implement `greedy_search`, which just returns the most likely next token. If multiple tokens are equally likely, break the tie by returning the smallest token.

Why not break ties randomly? It's nice that greedy search is deterministic, and also nice to not have special code for a case that rarely occurs (floats are rarely exactly equal).

Tip: the type checker doesn't know the return type of `item()` is int, but you can assert that it really is an int and this will make the type checker happy.

```python
def greedy_search(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    pass

prompt = "Jingle bells, jingle bells, jingle all the way"
print("Greedy decoding with prompt: ", prompt)
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Greedy decoding a second time (should be deterministic): ")
output = sample_tokens(gpt, tokenizer, prompt, max_tokens_generated=8, temperature=0.0)
print(f"Your model said: {output}")
expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
assert output == expected

print("Tests passed!")
```

## Sampling with `Categorical`

PyTorch provides a [`distributions` package](https://pytorch.org/docs/stable/distributions.html#distribution) with a number of convenient methods for sampling from various distributions.

For now, we just need [`t.distributions.categorical.Categorical`](https://pytorch.org/docs/stable/distributions.html#categorical). Use this to implement `sample_basic`, which just samples from the provided logits (which may have already been modified by the temperature and frequency penalties).

Note that this will be slow since we aren't batching the samples, but don't worry about speed for now.

```python
def sample_basic(logits: t.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    pass

N = 20000
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_basic(unnormalized_logits) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, probs, atol=0.01, rtol=0)
print("Tests passed!")
```

### Temperature

Temperature sounds fancy, but it's literally just dividing the logits by the temperature.

```python
def apply_temperature(logits: t.Tensor, temperature: float) -> t.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    pass

logits = t.tensor([1, 2]).log()
cold_logits = apply_temperature(logits, 0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)
hot_logits = apply_temperature(logits, 1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)
print("Tests passed!")
```
""")

    with st.expander("Question - what is the limit of applying 'sample_basic' after adjusting with temperature, when temperature goes to zero? How about when temperature goes to infinity?"):
        st.markdown("""
The limit when temperature goes to zero is greedy search (because dividing by a small number makes the logits very big, in other words the difference between the maximum logit one and all the others will grow). 

The limit when temperature goes to infinity is uniform random sampling over all words (because all logits will be pushed towards zero).")
""")

    st.markdown(r"""
### Frequency Penalty

The frequency penalty is simple as well: count the number of occurrences of each token, then subtract `freq_penalty` for each occurrence. Hint: use `t.bincount` (documentation [here](https://pytorch.org/docs/stable/generated/torch.bincount.html)) to do this in a vectorized way.""")

    with st.expander("""Help - I'm getting a RuntimeError; my tensor sizes don't match."""):
        st.markdown("""
Look at the documentation page for `t.bincount`. You might need to use the `minlength` argument - why?""")

    st.markdown(r"""
```python
def apply_freq_penalty(input_ids: t.Tensor, logits: t.Tensor, freq_penalty: float) -> t.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    pass

bieber_prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
input_ids = tokenizer.encode(bieber_prompt, return_tensors="pt").squeeze()
logits = t.ones(tokenizer.vocab_size)
penalized_logits = apply_freq_penalty(input_ids, logits, 2.0)
assert penalized_logits[5156].item() == -11, "Expected 6 occurrences of ' baby' with leading space"
assert penalized_logits[14801].item() == -5, "Expected 3 occurrences of ' Baby' with leading space"
print("Tests passed!")
```

### Sampling - Manual Testing

Run the below cell to get a sense for the `temperature` and `freq_penalty` arguments. Play with your own prompt and try other values.

Note: your model can generate newlines or non-printing characters, so calling `print` on generated text sometimes looks awkward on screen. You can call `repr` on the string before printing to have the string escaped nicely.

```python
N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(freq_penalty=100.0)),
    ("Negative freq penalty", dict(freq_penalty=-1.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]
for (name, kwargs) in cases:
    for i in range(N_RUNS):
        output = sample_tokens(gpt, tokenizer, your_prompt, max_tokens_generated=24, **kwargs)
        print(f"Sample {i} with: {name} ({kwargs}):")
        print(f"Your model said: {repr(output)}\n")
```

## Top-K Sampling

Conceptually, the steps in top-k sampling are:
- Find the `top_k` largest probabilities
- Set all other probabilities to zero
- Normalize and sample

Your implementation should stay in log-space throughout (don't exponentiate to obtain probabilities). This means you don't actually need to worry about normalizing, because `Categorical` accepts unnormalised logits.
""")

    with st.expander("Help - I don't know what function I should use for finding the top k."):
        st.markdown("Use [`t.topk`](https://pytorch.org/docs/stable/generated/torch.topk.html).")

    st.markdown("""
```python
def sample_top_k(logits: t.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    pass

k = 3
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_k(unnormalized_logits, k) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[:-k] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0)
print("Tests passed!")
```

### Top-K Sampling - Example
The [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) famously included an example prompt about unicorns. Now it's your turn to see just how cherry picked this example was.

The paper claims they used `top_k=40` and best of 10 samples.

```python
your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")
```

## Top-p aka Nucleus Sampling

Conceptually, in top-p sampling we:

- Sort the probabilities from largest to smallest
- Find the cutoff point where the cumulative probability first equals or exceeds `top_p`. We do the cutoff inclusively, keeping the first probability above the threshold.
- If the number of kept probabilities is less than `min_tokens_to_keep`, keep that many tokens instead.
- Set all other probabilities to zero
- Normalize and sample

Optionally, refer to the paper [The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf) for some comparison of different methods.


""")

    with st.expander("Help - I'm confused about how nucleus sampling works."):
        st.markdown("""The basic idea is that we choose the most likely words, up until the total probability of words we've chosen crosses some threshold. Then we sample from those chosen words based on their logits.
    
For instance, if our probabilities were `(0.4, 0.3, 0.2, 0.1)` and our cutoff was `top_p=0.8`, then we'd sample from the first three elements (because their total probability is `0.9` which is over the threshold, but the first two only have a total prob of `0.7` which is under the threshold). Once we've chosen to sample from those three, we would renormalise them by dividing by their sum (so the probabilities we use when sampling are `(4/9, 3/9, 2/9)`.""")

    with st.expander("Help - I'm stuck on how to implement this function."):
        st.markdown("""First, sort the logits using the `sort(descending=True)` method (this returns values and indices). Then you can get `cumulative_probs` by applying softmax to these logits and taking the cumsum. Then, you can decide how many probabilities to keep by using the `t.searchsorted` function.
    
Once you've decided which probabilities to keep, it's easiest to sample from them using the original logits (you should have preserved the indices when you called `logits.sort`). This way, you don't need to worry about renormalising like you would if you were using probabiliities.""")

    st.markdown("""
```python
def sample_top_p(logits: t.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    pass

N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.5) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p of 0.5 or lower should only return token 2: ", counts)
assert counts[0] == 0 and counts[1] == 0

N = 2000
unnormalized_logits = t.tensor([0.2, 0.3, 0.5]).log() + 2.3456
samples = t.tensor([sample_top_p(unnormalized_logits, 0.50001) for _ in range(N)])
counts = t.bincount(samples, minlength=len(unnormalized_logits)) / N
print("top_p in (0.5, 0.8] should return tokens 1 and 2: ", counts)
assert counts[0] == 0

N = 4000
top_p = 0.71
probs = t.linspace(0, 0.4, 5)
unnormalized_logits = probs.log() + 1.2345
samples = t.tensor([sample_top_p(unnormalized_logits, top_p) for _ in range(N)])
counts = t.bincount(samples, minlength=len(probs)) / N
expected = probs.clone()
expected[0:2] = 0
expected /= expected.sum()
print("Checking empirical frequencies (try to increase N if this test fails): ", counts)
t.testing.assert_close(counts, expected, atol=0.01, rtol=0.0)

print("All tests passed!")
```

### Top-p Sampling - Example

```python
your_prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
output = sample_tokens(gpt, tokenizer, your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=64)
print(f"Your model said: {repr(output)}")
```
""")

def section2():
    st.sidebar.markdown("""
## Table of Contents

<ul class="contents">
    <li><a class="contents-el" href="#defining-a-dataset">Defining a dataset</a></li>
    <li><a class="contents-el" href="#defining-a-tokenizer">Defining a tokenizer</a></li>
    <li><a class="contents-el" href="#final-notes">Final notes</a></li>
</ul>
""", unsafe_allow_html=True)

    st.markdown(r"""
# Training your transformer on Shakespeare

Now that we've discussed sampling, we can proceed to train our transformer on the Shakespeare corpus!

## Defining a dataset

You can access the complete works of Shakespeare at [this link](https://www.gutenberg.org/files/100/100-0.txt). You should tokenize the corpus as recommended by Jacob, using `re.split(r"\b", ...)`. If you're unfamiliar with how to use Python regular expressions, you might want to read [this w3schools tutorial](https://www.w3schools.com/python/python_regex.asp)

When creating a dataset (and dataloader) from the Shakespeare corpus, remember to tokenize by splitting at `"\b"`. You can follow most of the same steps as you did for your reversed digits dataset, although your dataset here will be a bit more complicated.
""")

    with st.expander("Help - I'm not sure how to convert my words into token ids."):
        st.markdown("""Once you tokenizer the corpus, you can define a list of all the unique tokens in the dataset. You can then sort them alphabetically, and tokenize by associating each token with its position in this sorted list.""")

    st.markdown("""
Training on the entire corpus might take a few hours. I found that you can get decent results (see below) from just training on 1% of the corpus, which only takes a few minutes. Your mileage may vary.

## Defining a tokenizer

If you want to use the functions from the previous section to sample tokens from your model, you'll need to construct a tokenizer from your dataset. As a minumum, your tokenizer should have methods `encode` and `decode`, and a `model_max_length` property. Here is a template you can use:

```python
class WordsTokenizer():
    model_max_length: int

    def __init__(self, wordsdataset: WordsDataset):
        pass

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.
        
        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        pass

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        pass
```

Note that if you want to call your tokenizer like `tokenizer(input_text)`, you'll need to define the special function `__call__`.

## Final notes

This task is meant to be open-ended and challenging, and can go wrong in one of many different ways! If you're using VSCode, this is probably a good time to get familiar with the [debugger](https://code.visualstudio.com/docs/editor/debugging) if you haven't already! It can help you avoid many a ... winter of discontent.

When choosing config parameters, try starting with the same ones described in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Next week, we'll look at [wandb](https://wandb.ai/site), and how it gives you a more systematic way of choosing parameters.

Once you succeed, you'll be able to create riveting output like this (produced after about 10 mins of training):

```python
initial_text = "turn down for what"

# Defining my transformer
model = DecoderOnlyTransformer(config).to(device).train()
# Defining my own tokenizer as function of trainset (see bullet point above)
tokenizer = WordsTokenizer(trainset)

text_output = sample_tokens(model, tokenizer, initial_text, max_tokens_generated=100, temperature=1.0, top_k=10)

print(text_output)

# turn down for what you do you think,
# That take the last, of many, which is so much I
# As this blows along than my life thou say’st, which makes thy hand,
# Thou wilt be given, or more
# Entitled in thy great world’s fresh blood will,
# To answer th’ alluring countenance, beauty 
```

Also, if you manage to generate text on the Shakespeare corpus, you might want to try other sources. When I described this task to my sister, she asked me what would happen if I trained the transformer on the set of all Taylor Swift songs - I haven't had time to try this, but bonus points if anyone gets good output by doing this.
""")

def section_caching():
    st.markdown(r"""
# Caching

Coming soon!
""")

func_page_list = [
    (section_home, "🏠 Home"), 
    (section_training, "1️⃣ Training"),
    (section_sampling, "2️⃣ Sampling"),
    (section_caching, "3️⃣ Caching"),
]

func_list = [func for func, page in func_page_list]
page_list = [page for func, page in func_page_list]

page_dict = {page: idx for idx, (func, page) in enumerate(func_page_list)}

def page():
    with st.sidebar:
        radio = st.radio("Section", page_list)
        st.markdown("---")
    func_list[page_dict[radio]]()

page()
