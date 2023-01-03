import torch as t
from tokenizers import models, Tokenizer, trainers, processors
from transformers import PreTrainedTokenizerFast
from typing import List, Union

MAIN = __name__ == "__main__"

class ToyTokenizer(PreTrainedTokenizerFast):

    def __init__(
        self,
        vocab: Union[str, List[str], int, List[int]],
        pad: bool = True,
        cls: bool = True,
        sep: bool = True,
        unk: bool = False,
        mask: bool = False,
    ):
        """Creates a tokenizer from a list of characters, and a list of special tokens.

        Args:
            vocab_list (Optional[str, List[str]]):
                if str, then it is a string of characters to use as the vocabulary.
                if List[str], then it is a list of characters to use as the vocabulary.
                if List[int], then it is a list of integers to use as the vocabulary.
                if int, then it is the size of the vocabulary to generate (vocab assumed to be integers 0, 1, ..., vocab-1).

            pad, cls, sep, unk, mask (bool):
                whether to include these special tokens in the vocabulary.
                recall that [CLS] and [SEP] play the role of start and end tokens, respectively.
        """

        self.uses_pad = pad
        self.uses_cls = cls
        self.uses_sep = sep
        self.uses_unk = unk
        self.uses_mask = mask

        if isinstance(vocab, str):
            vocab = list(vocab)
        elif isinstance(vocab, int):
            vocab = list(range(vocab))
        self.originally_int = isinstance(vocab[0], int)
        if self.originally_int:
            vocab = map(str, vocab)

        special_tokens = {}
        for token_name in ["pad", "cls", "sep", "unk", "mask"]:
            if getattr(self, f"uses_{token_name}"):
                special_tokens[f"{token_name}_token"] = f"[{token_name.upper()}]"

        # trainer = trainers.WordPieceTrainer(vocab_size=100, special_tokens=list(special_tokens.values()))
        tokenizer_raw = Tokenizer(models.Unigram([(char, 0) for char in vocab]))
        tokenizer_raw.add_special_tokens(list(special_tokens.values()))
        tokenizer_raw.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[(k, v) for k, v in tokenizer_raw.get_vocab().items() if k in ["[CLS]", "[SEP]"]],
        )

        self.tokenizer = tokenizer_raw
        super().__init__(tokenizer_object=tokenizer_raw, **special_tokens)

    # Started writing this to deal with cases where vocab is ints, but proved not to be worth the effort

    # def decode(self, *args, **kwargs):
    #     assert "skip_special_tokens" not in kwargs or kwargs["skip_special_tokens"] is not False, "skip_special_tokens=False is not supported."
    #     kwargs["skip_special_tokens"] = True
    #     out = super().decode(*args, **kwargs)
    #     out_split = out.split(" ")
    #     if self.originally_int:
    #         return list(map(int, out_split))
    #     else:
    #         return "".join(out_split)

    # def batch_decode(self, *args, **kwargs):
    #     assert "skip_special_tokens" not in kwargs or kwargs["skip_special_tokens"] is not False, "skip_special_tokens=False is not supported."
    #     kwargs["skip_special_tokens"] = True
    #     return super().batch_decode(*args, **kwargs)





# %%

# EXAMPLE 1: vocab is a string of characters

if MAIN:

    vocab = "()"
    tokenizer = ToyTokenizer(vocab)

    print(tokenizer(["()", "()()"], padding=True))
    # {
    #    'input_ids': [[3, 0, 1, 4, 2, 2], [3, 0, 1, 0, 1, 4]], 
    #    'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 
    #    'attention_mask': [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]]
    # }

    print(tokenizer.vocab)
    # {'(': 0, ')': 1, '[CLS]': 3, '[SEP]': 4, '[PAD]': 2, }

    print(tokenizer.decode([3, 0, 1, 4, 2, 2])) # Note - this defaults to skip_special_tokens=True, unlike standard behaviour
    # ()

    print(tokenizer.convert_tokens_to_ids("("))
    # 0

    print(tokenizer.convert_ids_to_tokens(0))
    # (





# %%

# EXAMPLE 2: vocab is a range of integers
# Note - you still use this by feeding in strings, not integers, e.g. "012" not [0, 1, 2]
# I started writing code to fix this but realising it wasn't worth the effort

if MAIN:

    vocab = 5
    tokenizer = ToyTokenizer(vocab)

    tokenizer([0, 1, 2, 3, 4], padding="max_length", add_special_tokens=True, max_length=10, return_tensors="pt")
    # {
    #     'input_ids': tensor([[6, 0, 1, 2, 3, 4, 7, 5, 5, 5]]), 
    #     'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
    #     'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    # }

    tokenizer.decode([6, 0, 1, 2, 3, 4, 7, 5, 5, 5])
    # '[CLS] 0 1 2 3 4 [SEP] [PAD] [PAD] [PAD]'

    tokenizer.decode([6, 0, 1, 2, 3, 4, 7, 5, 5, 5])
    # [0, 1, 2, 3, 4]

    tokenizer.batch_decode([[6, 0, 1, 2, 3, 4, 7, 5, 5, 5], [6, 0, 1, 2, 3, 4, 7, 5, 5, 5]])
    # [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
