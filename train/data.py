import os
import numpy as np
import torch


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenmes
        data_root = "../data/edu_fineweb10B"
        # data_root = "../data/edu_fineweb10B_custom_tokenizer"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        assert len(self.shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(self.shards)} shards for split {split}")

        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        buf = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]
        x = (buf[:-1]).view(self.B, self.T)  # inputs
        y = (buf[1:]).view(self.B, self.T)  # targets

        # advance the position in the tensor
        self.current_position += self.B * self.T * self.num_processes

        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (self.B * self.T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y
