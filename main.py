import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data import load_data, make_tasks, print_tasks, Tokenizer
from models import Detector, make_generator, get_features, HUMAN_KIND, llm_kind
from train import run_experiment, print_comparison, average_results, save_json


METHODS = ["static", "finetune", "full_retrain", "cgl"]


def get_args():
    parser = argparse.ArgumentParser(description="CGL for AI-generated text detection")
    parser.add_argument("--mode", required=True, choices=["check", "train", "compare", "generate", "detect", "demo"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--generator", dest="generator_type", choices=["diffusion", "gaussian"])
    parser.add_argument("--seeds", type=int, nargs="+", help="for compare: run every method with each seed")
    parser.add_argument("--data_folder")
    parser.add_argument("--output", default="outputs", help="folder to save results in")
    parser.add_argument("--device", help="auto, cpu or cuda")
    # hyperparameters
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_per_task", type=int)
    parser.add_argument("--max_words", type=int)
    parser.add_argument("--replay_ratio", type=float)
    parser.add_argument("--ewc_lambda", type=float)
    parser.add_argument("--encoder_lr_scale", type=float)
    parser.add_argument("--gen_steps", type=int)
    parser.add_argument("--diffusion_steps", type=int)
    parser.add_argument("--seed", type=int)
    # for generate / detect
    parser.add_argument("--checkpoint")
    parser.add_argument("--n", type=int, default=20, help="how many fake samples to make of each kind")
    parser.add_argument("--llm", help="for generate: only make fake samples of this LLM (e.g. llama)")
    parser.add_argument("--text")
    return parser.parse_args()


def load_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # anything given on the command line replaces the value in config.yaml
    for key in ["method", "generator_type", "data_folder", "device", "max_words", "epochs", "lr", "batch_size", "max_per_task",
                "replay_ratio", "ewc_lambda", "encoder_lr_scale", "gen_steps", "diffusion_steps", "seed"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(config):
    data = load_data(config["data_folder"])
    families = config["families"]
    train_tasks = make_tasks(data["train"], families, config["max_per_task"], config["seed"])
    test_tasks = make_tasks(data["test"], families, config["max_test_per_task"], config["seed"])
    valid_tasks = None
    if "valid" in data:
        valid_tasks = make_tasks(data["valid"], families, config["max_test_per_task"], config["seed"])

    # skip LLMs that have no texts in this dataset (e.g. in the small demo data)
    keep = [i for i in range(len(train_tasks)) if len(train_tasks[i]) > 0 and len(test_tasks[i]) > 0]
    names = list(families.keys())
    if len(keep) < len(names):
        print("No data for:", [names[i] for i in range(len(names)) if i not in keep], "- skipping them")
        config["families"] = {names[i]: families[names[i]] for i in keep}
        train_tasks = [train_tasks[i] for i in keep]
        test_tasks = [test_tasks[i] for i in keep]
        if valid_tasks:
            valid_tasks = [valid_tasks[i] for i in keep]
    # one word list for everything (built from the training texts)
    all_texts = []
    for task in train_tasks:
        all_texts += list(task["text"])
    tokenizer = Tokenizer.build(all_texts, config["vocab_size"])
    return train_tasks, valid_tasks, test_tasks, tokenizer


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer = Tokenizer(checkpoint["words"])
    detector = Detector(len(tokenizer), config["embed_size"], config["num_heads"], config["num_layers"],
                        max_words=config["max_words"])
    detector.load_state_dict(checkpoint["detector"])
    generator = None
    if checkpoint["generator"] is not None:
        generator = make_generator(config)
        generator.load_state_dict(checkpoint["generator"])
    return config, tokenizer, detector, generator, checkpoint


def generate_samples(args, config, tokenizer, detector, generator, checkpoint):
    """
    Generation workflow: the generator makes fake feature vectors of human text and of every LLM it knows.
    A feature vector is just 128 numbers, so to show what a fake sample "means" we find the REAL MAGE
    test text whose feature vector is closest to it. If the generator works, a fake "llama" sample should
    be closest to a real LLaMA text.
    """
    families = checkpoint["families"]
    learned = families[:checkpoint["task"]]                 # the LLMs this generator knows
    if args.llm:
        if args.llm not in learned:
            print("The generator doesn't know", args.llm, "- choose one of:", ", ".join(learned))
            return
        learned = [args.llm]
    print(f"{config['generator_type']} generator after task {checkpoint['task']} (knows: {', '.join(families[:checkpoint['task']])})\n")

    # real test texts of the same LLMs, to compare with
    data_folder = args.data_folder or config["data_folder"]
    test = load_data(data_folder)["test"]
    real_tasks = make_tasks(test, {name: config["families"][name] for name in learned}, 200, config["seed"])
    real_feats, real_names, real_texts = [], [], []
    for name, task in zip(learned, real_tasks):
        f, y = get_features(detector, task, tokenizer, config["max_words"], "cpu")
        real_feats.append(f)
        real_names += [name if label == 1 else "human" for label in y.tolist()]
        real_texts += list(task["text"])
    real_feats = torch.cat(real_feats)

    kinds = [("human", HUMAN_KIND)] + [(name, llm_kind(families.index(name))) for name in learned]
    output, matches = [], {}
    for name, kind in kinds:
        feats = generator.write(args.n, kind)
        probs = detector.predict_features(feats).tolist()
        # closest real text (cosine similarity)
        similarity = F.normalize(feats, dim=1) @ F.normalize(real_feats, dim=1).T
        closest = similarity.argmax(dim=1).tolist()
        same_llm = sum(real_names[c] == name for c in closest) / len(closest)
        same_label = sum((real_names[c] == "human") == (name == "human") for c in closest) / len(closest)
        matches[name] = {"closest_real_text_has_same_label": same_label, "closest_real_text_is_same_llm": same_llm}
        label_word = "human" if name == "human" else "AI"
        print(f"=== fake '{name}' samples: closest real text is {label_word}-written for {same_label * 100:.0f}% of them"
              + ("" if name == "human" else f", and from {name} itself for {same_llm * 100:.0f}%") + " ===")
        for p, c in zip(probs[:3], closest[:3]):
            print(f"  detector says AI with p={p:.2f} | closest real text ({real_names[c]}): {real_texts[c][:110]!r}")
        for p, c in zip(probs, closest):
            output.append({"asked_for": name, "detector_p_ai": round(p, 3),
                           "closest_real_text_is_from": real_names[c], "closest_real_text": real_texts[c][:300]})
        print()

    out_folder = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "generated")
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "generated_samples.json"), "w") as f:
        json.dump({"closest_real_text_matches": matches, "samples": output}, f, indent=2)
    print("Saved the samples to", out_folder)


def main():
    args = get_args()

    if args.mode in ["generate", "detect"]:
        if not args.checkpoint:
            print("Please give a --checkpoint (e.g. outputs/cgl/checkpoints/after_task11.pt)")
            return
        config, tokenizer, detector, generator, checkpoint = load_checkpoint(args.checkpoint)

        if args.mode == "detect":
            text = args.text or input("Paste a text: ")
            ids = torch.tensor([tokenizer.encode(text, config["max_words"])])
            p = detector.predict(ids).item()
            print(f"Probability AI-generated: {p:.3f}")
            print("Verdict:", "AI-generated" if p > 0.5 else "human-written")
            return

        if generator is None:
            print("This checkpoint has no generator (only the cgl method trains one).")
            return
        generate_samples(args, config, tokenizer, detector, generator, checkpoint)
        return

    # check / train / compare need the data
    config = load_config(args)
    set_seed(config["seed"])
    train_tasks, valid_tasks, test_tasks, tokenizer = prepare_data(config)
    family_names = list(config["families"].keys())
    print_tasks(train_tasks, family_names)
    print("Vocabulary size:", len(tokenizer))

    if args.mode == "check":
        return

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "config_used.yaml"), "w") as f:
        yaml.dump(config, f)

    if args.mode == "train":
        method = config["method"]
        results = run_experiment(config, method, train_tasks, valid_tasks, test_tasks, tokenizer,
                                 os.path.join(args.output, method))
        print_comparison([average_results([results])], os.path.join(args.output, method))
        print("\nResults saved in", os.path.join(args.output, method))

    if args.mode == "compare":
        seeds = args.seeds or [config["seed"]]
        runs = {m: [] for m in METHODS}
        for seed in seeds:
            for method in METHODS:
                print(f"\n\n******** Running method: {method} (seed {seed}) ********")
                c = dict(config)
                c["seed"] = seed
                set_seed(seed)
                folder = os.path.join(args.output, method) if len(seeds) == 1 else \
                    os.path.join(args.output, f"seed{seed}", method)
                runs[method].append(run_experiment(c, method, train_tasks, valid_tasks, test_tasks, tokenizer, folder))

        # average over the seeds (with 1 seed this just copies the numbers)
        all_results = [average_results(runs[m]) for m in METHODS]
        save_json(all_results, os.path.join(args.output, "results.json"))
        print_comparison(all_results, args.output)


if __name__ == "__main__":
    main()