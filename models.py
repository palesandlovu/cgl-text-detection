# All the "learning" parts of the project

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_batches, make_samples


# the detector
# ====================================================================
# A small transformer that reads a text and says if it is AI-generated or human-written.


class Detector(nn.Module):
    def __init__(self, vocab_size, embed_size=128, num_heads=4, num_layers=2, max_words=512):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_words, embed_size)   # learns where each word is
        layer = nn.TransformerEncoderLayer(embed_size, num_heads, dim_feedforward=embed_size * 2,
                                           dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(embed_size, embed_size), nn.ReLU(), nn.Linear(embed_size, 1))

    def features(self, ids):
        #Text (word ids) -> one feature vector per text.
        padding = ids == 0
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.word_embedding(ids) + self.position_embedding(positions)
        x = self.transformer(x, src_key_padding_mask=padding)
        # average all the word vectors (but not the padding) to get one vector per text
        not_padding = (~padding).unsqueeze(-1).float()
        return (x * not_padding).sum(dim=1) / not_padding.sum(dim=1).clamp(min=1)

    def forward(self, ids):
        return self.head(self.features(ids)).squeeze(-1)    # one number: > 0 means AI, < 0 means human

    def predict(self, ids):
        #Returns the probability that each text is AI-generated.
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            return torch.sigmoid(self.forward(ids.to(device))).cpu()

    def predict_features(self, feats):
        #Same as predict, but starting from feature vectors (used for the fake ones).
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            return torch.sigmoid(self.head(feats.to(device)).squeeze(-1)).cpu()


def get_features(detector, df, tokenizer, max_words, device):
    #Feature vectors and labels of all texts in a table.
    feats, labels = [], []
    detector.eval()
    with torch.no_grad():
        for ids, y, _, _ in make_batches(make_samples(df, tokenizer, max_words), 64, shuffle=False):
            feats.append(detector.features(ids.to(device)).float().cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


# the generators
# ====================================================================
# The generators learn what the detector's feature vectors look like for each "kind" of text:

HUMAN_KIND = 0
MAX_KINDS = 64


def llm_kind(k):
    return k + 1


class DiffusionGenerator(nn.Module):
    """
    A small diffusion model (DDPM, Ho et al. 2020) - the same idea as DDGR (Gao & Liu 2023),
    but it makes feature vectors instead of images.
    """

    def __init__(self, feature_size=128, hidden_size=256, steps=50):
        super().__init__()
        self.steps = steps
        noise = torch.linspace(1e-4, 0.2, steps)                        # how much noise is added at each step
        # (after the last step almost nothing of the original is left - it is pure noise,
        #  which is exactly where writing a new sample starts)
        self.register_buffer("noise", noise)
        self.register_buffer("keep", torch.cumprod(1 - noise, dim=0))   # how much of the original is left
        self.kind_embedding = nn.Embedding(MAX_KINDS, 64)               # tells it which kind to make
        self.step_embedding = nn.Embedding(steps, 64)                   # tells it which step it is at
        self.network = nn.Sequential(nn.Linear(feature_size + 128, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, hidden_size), nn.SiLU(),
                                     nn.Linear(hidden_size, feature_size))
        # the features are scaled to average 0 and spread 1 before the diffusion (works better)
        self.register_buffer("mean", torch.zeros(feature_size))
        self.register_buffer("std", torch.ones(feature_size))
        self.register_buffer("scaled", torch.tensor(False))

    def guess_noise(self, x, step, kind):
        return self.network(torch.cat([x, self.step_embedding(step), self.kind_embedding(kind)], dim=-1))

    def learn(self, feats, kinds, config, log):
        device = self.mean.device
        if not self.scaled:            # remember the scale of the first LLM's features
            self.mean.copy_(feats.mean(0).to(device))
            self.std.copy_((feats.std(0) + 1e-3).to(device))
            self.scaled.fill_(True)
        x0_all = (feats.to(device) - self.mean) / self.std
        kinds = kinds.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=config["gen_lr"])
        # the learning rate slowly goes down to 0 (cosine schedule) - this made the samples much better
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["gen_steps"])
        self.train()
        total = 0
        # the network is tiny, so we train for a fixed number of steps (random batches of 256)
        for step_number in range(1, config["gen_steps"] + 1):
            idx = torch.randint(0, len(x0_all), (256,), device=device)
            x0, kind = x0_all[idx], kinds[idx]
            step = torch.randint(0, self.steps, (len(idx),), device=device)
            noise = torch.randn_like(x0)
            keep = self.keep[step].unsqueeze(-1)
            noisy = keep.sqrt() * x0 + (1 - keep).sqrt() * noise          # forward process
            loss = F.mse_loss(self.guess_noise(noisy, step, kind), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item()
            if step_number % 1000 == 0:
                log(f"    generator step {step_number}/{config['gen_steps']}: loss {total / 1000:.4f}")
                total = 0

    def write(self, n, kind):
        #Make n fake feature vectors of one kind (reverse process).
        self.eval()
        device = self.mean.device
        kinds = torch.full((n,), kind, dtype=torch.long, device=device)
        x = torch.randn(n, len(self.mean), device=device)
        with torch.no_grad():
            for t in range(self.steps - 1, -1, -1):
                step = torch.full((n,), t, dtype=torch.long, device=device)
                noise, keep = self.noise[t], self.keep[t]
                x = (x - noise / (1 - keep).sqrt() * self.guess_noise(x, step, kinds)) / (1 - noise).sqrt()
                if t > 0:
                    x = x + noise.sqrt() * torch.randn_like(x)
        return (x * self.std + self.mean).cpu()


class GaussianGenerator(nn.Module):
    """
    The simplest possible generator, to compare with the diffusion one:
    it only remembers the average and spread of every number in the feature vector, for each kind.
    """

    def __init__(self, feature_size=128):
        super().__init__()
        self.register_buffer("means", torch.zeros(MAX_KINDS, feature_size))
        self.register_buffer("stds", torch.ones(MAX_KINDS, feature_size))

    def learn(self, feats, kinds, config, log):
        for kind in kinds.unique().tolist():
            self.means[kind] = feats[kinds == kind].mean(0).to(self.means.device)
            self.stds[kind] = feats[kinds == kind].std(0).to(self.means.device)
        log(f"    gaussian generator: remembered the average and spread of {len(kinds.unique())} kinds")

    def write(self, n, kind):
        noise = torch.randn(n, self.means.shape[1], device=self.means.device)
        return (self.means[kind] + self.stds[kind] * noise).cpu()


def make_generator(config):
    if config["generator_type"] == "diffusion":
        return DiffusionGenerator(config["embed_size"], config["gen_hidden_size"], config["diffusion_steps"])
    return GaussianGenerator(config["embed_size"])


def size_in_mb(model):
    total = 0
    for p in list(model.parameters()) + list(model.buffers()):
        total += p.numel() * p.element_size()
    return total / 1e6


# generative feature replay
# ====================================================================
# Generative replay 

"""Instead of saving all the old texts, keep a small generator that learnt what the detector's
feature vectors of the old LLMs looked like. When a new LLM comes out, the generator makes fake
feature vectors of every old LLM,"""


def train_generator(generator, detector, task_df, tokenizer, config, device, log, task_number):
    """Teach the generator the features of the new LLM (and remind it of the old LLMs)."""
    start = time.time()
    feats, labels = get_features(detector, task_df, tokenizer, config["max_words"], device)
    kinds = torch.where(labels == 1, llm_kind(task_number), HUMAN_KIND)

    # self-replay: add the generator's own fake samples of every old LLM
    if task_number > 0 and config["generator_type"] == "diffusion":
        per_llm = max(int(len(feats) * config["replay_ratio"]) // task_number, 1)
        old_feats, old_kinds = [], []
        for k in range(task_number):
            old_feats.append(generator.write(per_llm, llm_kind(k)))
            old_kinds.append(torch.full((per_llm,), llm_kind(k)))
        feats = torch.cat([feats] + old_feats)
        kinds = torch.cat([kinds] + old_kinds)

    generator.learn(feats, kinds, config, log)
    return time.time() - start


def make_fake_features(generator, old_detector, n_wanted, n_old_llms):
    """
    Make n_wanted fake feature vectors: half human, half split equally between the old LLMs.
    Returns (feature vectors, targets, how often the old detector agrees with the kind asked for)
    """
    feats = [generator.write(n_wanted // 2, HUMAN_KIND)]
    labels = [0] * (n_wanted // 2)
    per_llm = max(n_wanted // 2 // n_old_llms, 1)
    for k in range(n_old_llms):
        feats.append(generator.write(per_llm, llm_kind(k)))
        labels += [1] * per_llm
    feats = torch.cat(feats)
    targets = old_detector.predict_features(feats)    # the old detector's probability = what to learn
    labels = torch.tensor(labels)
    agree = ((targets > 0.5).long() == labels).float().mean().item()
    return feats, targets, agree


# EWC (Elastic Weight Consolidation, Kirkpatrick et al. 2017)
# ====================================================================
# After learning an LLM, check which weights were important for it.


class EWC:
    def __init__(self, model, strength):
        self.model = model
        self.strength = strength
        self.importance = None      # how important each weight is (Fisher information)
        self.old_weights = None

    def remember(self, samples, device, n_batches=30):
        # Call this after finishing a task.
        if self.strength == 0:
            return
        importance = {}
        for name, w in self.model.named_parameters():
            importance[name] = torch.zeros_like(w)

        self.model.eval()
        count = 0
        for ids, labels, _, _ in make_batches(samples, 32):
            self.model.zero_grad()
            out = self.model(ids.to(device))
            loss = F.binary_cross_entropy_with_logits(out, labels.float().to(device))
            loss.backward()
            for name, w in self.model.named_parameters():
                if w.grad is not None:
                    importance[name] += w.grad.detach() ** 2   # big gradient = important weight
            count += 1
            if count == n_batches:
                break
        for name in importance:
            importance[name] /= count

        # keep (most of) the importance from older tasks too
        if self.importance is not None:
            for name in importance:
                importance[name] += 0.9 * self.importance[name].float()
        self.importance = {name: v.half() for name, v in importance.items()}
        self.old_weights = {name: w.detach().half() for name, w in self.model.named_parameters()}
        self.model.zero_grad()

    def memory_mb(self):
        # How much memory EWC needs to keep between tasks.
        if self.importance is None:
            return 0.0
        total = 0
        for name in self.importance:
            total += self.importance[name].numel() * 2 + self.old_weights[name].numel() * 2
        return total / 1e6

    def penalty(self):
        if self.importance is None or self.strength == 0:
            return 0.0
        total = 0.0
        for name, w in self.model.named_parameters():
            total += (self.importance[name].float() * (w - self.old_weights[name].float()) ** 2).sum()
        return self.strength * total
