import os
from tqdm import tqdm

from datasets import load_dataset
import sentencepiece as spm


local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")  # total size: 9672101
dataset = dataset.select(range(0, 4000000))  #4m 

dataset_file = "samples.txt"
with open(dataset_file, "w") as f:
    for example in dataset:
        f.write(example["text"] + "\n")


model_prefix = "tokenizer"
vocab_size = 50304
model_type = "bpe"

print('started training')
spm.SentencePieceTrainer.train(
    input=dataset_file,
    model_prefix=model_prefix,
    model_type=model_type,
    vocab_size=vocab_size,
    normalization_rule_name='identity',  # turn off normalization
    remove_extra_whitespaces=False,
    # input_sentence_size=10000000000, # number of training examples: 10B
    # input_sentence_size=120000000, # number of training examples: 120m
    # input_sentence_size=50000000, # number of training examples: 50m
    input_sentence_size=20000000, # number of training examples: 20m
    max_sentence_length=1024,
    shuffle_input_sentence=True,
    character_coverage=1.0,
    byte_fallback=True,
    # merge_rules
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    # unk_id=0,
    # bos_id=1,
    # eos_id=2,
    # pad_id=-1,
    # unk_piece="<unk>",
    # bos_piece="<|begin_of_text|>",
    # eos_piece="<|end_of_text|>",
    num_threads=os.cpu_count(),
)

from transformers import LlamaTokenizer

tokenizer_path = "tokenizer.model"
tokenizer = LlamaTokenizer(tokenizer_path)

text = "A beginning is the time for taking the most delicate care that the balances are correct. This every sister of the Bene Gesserit knows. To begin your study of the life of Maud'Dib, then, take care that you first place him in his time: born in the 57th year of the Padishah Emperor, Shaddam IV. And take the most special care that you locate Maud'Dib in his place: the planet Arrakis. Do not be deceived by the fact that he was born on Caladan and lived his first fifteen years there. Arrakis, the planet known as Dune, is forever his place."

encoded = tokenizer.encode(text)
print(encoded)


# probably this is how you can convert the tokenizer to hf
# https://github.com/jzhang38/TinyLlama/blob/main/lit_gpt/tokenizer.py#L9-L29

##! this is still not working properly
# transforming the tokenizer into a pretrainedfasttokenizer by HF
# import sentencepiece as spm
# from transformers import PreTrainedTokenizerFast
# from tokenizers import SentencePieceBPETokenizer

# sp_model = spm.SentencePieceProcessor()
# sp_model.load("tokenizer.model")

# tokenizer = SentencePieceBPETokenizer()
# vocab = {sp_model.id_to_piece(id): id for id in range(sp_model.GetPieceSize())}
# tokenizer.add_tokens(list(vocab.keys()))
# special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
# tokenizer.add_special_tokens(special_tokens)
# fast_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object=tokenizer,
#     bos_token="<s>",
#     eos_token="</s>",
#     unk_token="<unk>",
#     pad_token="<pad>"
# )

# fast_tokenizer.save_pretrained("new-tokenizer-hf-fast")
