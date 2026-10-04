# Loading the MAGE dataset, splitting it into tasks (one task = one AI family)
# and turning text into numbers so the models can use it.


import os
import re
import random
from collections import Counter

import pandas as pd
import torch


def load_data(folder):
    """Load train, valid and test csv files from the MAGE folder."""
    data = {}
    for split in ["train", "valid", "test"]:
        path = os.path.join(folder, split + ".csv")
        if not os.path.exists(path):
            print("Could not find", path)
            continue
        print("Loading", path)
        df = pd.read_csv(path)
        df = df.dropna(subset=["text", "src"])
        df["text"] = df["text"].astype(str)
        # label: 0 if human, 1 if AI
        df["label"] = df["src"].apply(lambda s: 0 if "human" in s else 1)
        data[split] = df
    if "train" not in data or "test" not in data:
        raise Exception("I need at least train.csv and test.csv in " + folder)
    return data


def get_family(src, families):
    """Return the name of the AI family this text came from (or None)."""
    for family_name, keywords in families.items():
        for word in keywords:
            if word in src:
                return family_name
    return None


def make_tasks(df, families, max_per_task, seed=42):
    #Split the data into one smaller table per AI family.
    df = df.copy()
    df["family"] = df["src"].apply(lambda s: get_family(s, families))
    df["topic"] = df["src"].apply(lambda s: s.split("_")[0])   # e.g. "xsum_human" -> "xsum"

    human = df[df["label"] == 0].sample(frac=1, random_state=seed)   # shuffle
    used_human = set()
    tasks = []

    for family_name in families:
        ai = df[(df["label"] == 1) & (df["family"] == family_name)]
        if len(ai) > max_per_task // 2:
            ai = ai.sample(n=max_per_task // 2, random_state=seed)

        # pick human texts with the same topics as the AI texts
        human_rows = []
        for topic, count in ai["topic"].value_counts().items():
            options = human[(human["topic"] == topic) & (~human.index.isin(used_human))]
            chosen = options.head(count)
            human_rows.append(chosen)
            used_human.update(chosen.index)
        human_part = pd.concat(human_rows) if human_rows else human.head(0)

        # if a topic ran out of human texts, fill up with human texts from any topic
        missing = len(ai) - len(human_part)
        if missing > 0:
            extra = human[~human.index.isin(used_human)].head(missing)
            used_human.update(extra.index)
            human_part = pd.concat([human_part, extra])

        task = pd.concat([ai, human_part]).sample(frac=1, random_state=seed)
        tasks.append(task[["text", "src", "label"]].reset_index(drop=True))
    return tasks


def print_tasks(tasks, family_names):
    print()
    print("Task  Family          #AI   #Human")
    for i, (name, task) in enumerate(zip(family_names, tasks)):
        n_ai = (task["label"] == 1).sum()
        n_human = (task["label"] == 0).sum()
        print(f"{i + 1:<6}{name:<16}{n_ai:<6}{n_human}")
    print()


# ---------------------------------------------------------------------
# Tokenizer: turns text into a list of word numbers
# ---------------------------------------------------------------------
# special words: <pad> = padding (fills up short texts), <unk> = a word that isn't in the word list
SPECIAL = ["<pad>", "<unk>"]


def split_words(text):
    return re.findall(r"[a-z0-9']+|[.,!?;:]", text.lower())


class Tokenizer:
    def __init__(self, words):
        self.words = words                                  # number -> word
        self.word_to_id = {w: i for i, w in enumerate(words)}  # word -> number

    @staticmethod
    def build(texts, vocab_size):
        counts = Counter()
        for text in texts:
            counts.update(split_words(text))
        most_common = [w for w, c in counts.most_common(vocab_size - len(SPECIAL))]
        return Tokenizer(SPECIAL + most_common)

    def encode(self, text, max_words):
        ids = [self.word_to_id.get(w, 1) for w in split_words(text)]   # 1 = <unk>
        ids = ids[:max_words]
        if len(ids) == 0:
            ids = [1]       # empty text -> just one <unk> so the model doesn't break
        return ids

    def decode(self, ids):
        words = []
        for i in ids:
            word = self.words[i]
            if word == "<end>":
                break
            if word not in SPECIAL:
                words.append(word)
        text = " ".join(words)
        text = re.sub(r" ([.,!?;:])", r"\1", text)      # "hello ." -> "hello."
        return text

    def __len__(self):
        return len(self.words)


# ---------------------------------------------------------------------
# Turning tables into lists the training code can use
# ---------------------------------------------------------------------
def make_samples(df, tokenizer, max_words):
    """
    Each sample is a list: [word ids, label, target, is_fake]
    - target is the number the detector should predict (same as label for real texts,
      for fake replay texts it's the old detector's guess)
    - is_fake = 1 for samples made by the generator
    """
    samples = []
    for text, label in zip(df["text"], df["label"]):
        samples.append([tokenizer.encode(text, max_words), label, float(label), 0])
    return samples


def make_batches(samples, batch_size, shuffle=True):
    """Split samples into batches and pad them so every text in a batch has the same length."""
    order = list(range(len(samples)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_size):
        batch = [samples[i] for i in order[start:start + batch_size]]
        longest = max(len(s[0]) for s in batch)
        ids = torch.zeros(len(batch), longest, dtype=torch.long)    # 0 = <pad>
        for row, s in enumerate(batch):
            ids[row, :len(s[0])] = torch.tensor(s[0])
        labels = torch.tensor([s[1] for s in batch])
        targets = torch.tensor([s[2] for s in batch], dtype=torch.float)
        is_fake = torch.tensor([s[3] for s in batch], dtype=torch.float)
        yield ids, labels, targets, is_fake

