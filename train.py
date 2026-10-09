# Training and testing

import os
import copy
import time
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import make_samples, make_batches
from models import Detector, make_generator, size_in_mb, train_generator, make_fake_features, EWC


# training loop


def get_device(config):
    if config["device"] == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config["device"])


def ram_used_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        return 0.0


def new_detector(config, tokenizer, device):
    return Detector(len(tokenizer), config["embed_size"], config["num_heads"], config["num_layers"],
                    max_words=config["max_words"]).to(device)


def train_detector(detector, samples, config, device, ewc, log, valid_samples=None, fake=None, slow_encoder=False):
    #Train the detector for a few epochs and print how it's going.
    
    if fake is not None:
        fake_feats, fake_targets = fake[0].to(device), fake[1].to(device)
    head_params = list(detector.head.parameters())
    encoder_params = [p for name, p in detector.named_parameters() if not name.startswith("head.")]
    encoder_lr = config["lr"] * (config["encoder_lr_scale"] if slow_encoder else 1.0)
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": encoder_lr},
                                   {"params": head_params, "lr": config["lr"]}])
    use_amp = config["use_amp"] and device.type == "cuda"     # mixed precision only on GPU
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_batches = (len(samples) + config["batch_size"] - 1) // config["batch_size"]

    for epoch in range(config["epochs"]):
        detector.train()
        start = time.time()
        total_loss, total_ewc, correct, seen = 0, 0, 0, 0

        for step, (ids, labels, targets, is_fake) in enumerate(make_batches(samples, config["batch_size"]), 1):
            ids, labels, targets = ids.to(device), labels.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = detector(ids)
            loss = F.binary_cross_entropy_with_logits(out.float(), targets)   # real texts: the true label
            if fake is not None:
                # replay: fake old features learn the old detector's probability
                pick = torch.randint(0, len(fake_feats), (len(ids),), device=device)
                fake_out = detector.head(fake_feats[pick]).squeeze(-1)
                loss = loss + F.binary_cross_entropy_with_logits(fake_out.float(), fake_targets[pick])
            ewc_loss = ewc.penalty() if ewc else 0.0

            scaler.scale(loss + ewc_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_ewc += float(ewc_loss) if not torch.is_tensor(ewc_loss) else ewc_loss.item()
            correct += ((out > 0).long() == labels).sum().item()
            seen += len(labels)

            if step % 20 == 0:
                speed = seen / (time.time() - start)
                log(f"    epoch {epoch + 1} batch {step}/{n_batches} | loss {total_loss / step:.4f} | "
                    f"ewc {total_ewc / step:.4f} | train acc {correct / seen:.3f} | {speed:.0f} texts/s")

        msg = (f"  epoch {epoch + 1}/{config['epochs']} done | loss {total_loss / n_batches:.4f} | "
               f"train acc {correct / seen:.3f}")
        if valid_samples:
            v = test_detector(detector, valid_samples)
            msg += f" | valid acc {v['accuracy']:.3f} | valid f1 {v['f1']:.3f}"
        msg += f" | {time.time() - start:.0f}s | RAM {ram_used_mb():.0f} MB"
        if device.type == "cuda":
            msg += f" | GPU memory {torch.cuda.max_memory_allocated() / 1e6:.0f} MB"
        log(msg)


def run_experiment(config, method, train_tasks, valid_tasks, test_tasks, tokenizer, folder, name=None):
    # method = how to trin (static, finetune, full_retrain, cgl, exemplar)
    # name   = what to call it in the results (e.g. "cgl_no_replay" is method "cgl" with replay_ratio 0)
    run_name = name or method
    os.makedirs(folder + "/checkpoints", exist_ok=True)
    log_file = open(folder + "/train_log.txt", "w")

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    torch.manual_seed(config["seed"])
    device = get_device(config)
    log(f"Method: {run_name} | seed: {config['seed']} | device: {device}" +
        (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    families = list(config["families"].keys())
    max_words = config["max_words"]
    test_sets = [make_samples(t, tokenizer, max_words) for t in test_tasks]
    valid_sets = [make_samples(t, tokenizer, max_words) for t in valid_tasks] if valid_tasks else [None] * len(families)

    detector = new_detector(config, tokenizer, device)
    generator = None
    ewc = None
    if method == "cgl":
        generator = make_generator(config).to(device)
        ewc = EWC(detector, config["ewc_lambda"])
        log(f"Detector size: {size_in_mb(detector):.2f} MB | {config['generator_type']} generator size: "
            f"{size_in_mb(generator):.2f} MB")

    accuracy_table, f1_table, auroc_table = [], [], []
    seen_accuracy_per_task = []
    next_accuracy, next_auroc = [], []       # accuracy on the NEXT LLM, before training on it
    task_info = []
    buffer = []                              # exemplar only: stored real texts as [word ids, label, target, is_fake, task]
    # size of every task's texts as 16-bit word ids (what the model actually reads), for a fair memory comparison
    word_id_mb = [sum(len(s[0]) * 2 for s in make_samples(t, tokenizer, max_words)) / 1e6 for t in train_tasks]

    for i, family in enumerate(families):
        log("")
        log("=" * 60)
        log(f"TASK {i + 1}/{len(families)}: new LLM = {family}")
        log("=" * 60)
        task_start = time.time()
        new_samples = make_samples(train_tasks[i], tokenizer, max_words)
        replay_info = {}
        train_samples = []
        fake = None

        # make the training data for this task
        if method == "static":
            if i == 0:
                train_samples = new_samples
            else:
                log("  static detector: not updated (it only ever learns the first LLM)")

        elif method == "finetune":
            train_samples = new_samples

        elif method == "full_retrain":
            detector = new_detector(config, tokenizer, device)      # start from zero again
            for j in range(i + 1):
                train_samples += make_samples(train_tasks[j], tokenizer, max_words)
            log(f"  retraining from scratch on all {len(train_samples)} real texts so far")

        elif method == "exemplar":
            # train on the new LLM + the stored real texts of the old LLMs
            train_samples = new_samples + [s[:4] for s in buffer]
            if buffer:
                log(f"  replaying {len(buffer)} stored real texts of {i} old LLMs")
            else:
                log("  first task, nothing stored yet")
            # then update the buffer: buffer_size texts in total, split equally over al LLMs so far
            per_task = config["buffer_size"] // (i + 1)
            new_buffer = []
            for j in range(i):
                new_buffer += [s for s in buffer if s[4] == j][:per_task]
            new_buffer += [s + [i] for s in new_samples[:per_task]]   # the task table is already shuffled
            buffer = new_buffer

        else:  # cgl
            train_samples = new_samples
            if i > 0 and config["replay_ratio"] > 0:
                # GENERATIVE REPLAY: old detector + generator = fake feature vectors of the old LLMs
                start = time.time()
                old_detector = copy.deepcopy(detector)
                n_fake = int(config["replay_ratio"] * len(new_samples))
                fake_feats, fake_targets, agree = make_fake_features(generator, old_detector, n_fake, n_old_llms=i)
                fake = (fake_feats, fake_targets)
                del old_detector
                replay_info = {"fake_made": len(fake_feats), "old_detector_agrees": agree,
                               "replay_time": time.time() - start}
                log(f"  replay: generator made {len(fake_feats)} fake feature vectors of {i} old LLMs + human "
                    f"in {replay_info['replay_time']:.1f}s (old detector agrees with {agree * 100:.0f}% of them)")
            else:
                log("  first task, nothing to replay yet")

        # train - updated detector 
        if train_samples:
            log(f"  training on {len(train_samples)} texts ({len(new_samples)} real from {family})")
            train_detector(detector, train_samples, config, device, ewc, log, valid_sets[i], fake,
                           slow_encoder=(method == "cgl" and i > 0))

        # cgl only: remember this LLM for next time 
        if method == "cgl":
            ewc.remember(new_samples, device)
            log(f"  teaching the {config['generator_type']} generator the new LLM...")
            train_generator(generator, detector, train_tasks[i], tokenizer, config, device, log, task_number=i)
        task_time = time.time() - task_start

        # what each method has to KEEP in memory to carry on with the next LLM
        if method == "full_retrain":
            stored_mb = sum(train_tasks[j]["text"].str.len().sum() for j in range(i + 1)) / 1e6   # all old texts
            stored_ids_mb = sum(word_id_mb[:i + 1])                  # the same texts stored as word ids
        elif method == "cgl":
            stored_mb = size_in_mb(generator) + ewc.memory_mb()      # generator + EWC, no old texts
            stored_ids_mb = stored_mb
        elif method == "exemplar":
            stored_mb = sum(len(s[0]) * 2 for s in buffer) / 1e6     # stored texts as 16-bit word ids
            stored_ids_mb = stored_mb
        else:
            stored_mb = stored_ids_mb = 0.0

        # test on the new LLM, all the old ones and the ones still to com 
        acc_row, f1_row, auroc_row = [], [], []
        log(f"\n  Results after task {i + 1}:")
        for j, name in enumerate(families):
            r = test_detector(detector, test_sets[j])
            acc_row.append(r["accuracy"])
            f1_row.append(r["f1"])
            auroc_row.append(r["auroc"])
            if j < i:
                tag = "old"
            elif j == i:
                tag = "NEW"
            else:
                tag = "unseen"
            log(f"    {tag:<7}{name:<14} accuracy {r['accuracy']:.3f} | f1 {r['f1']:.3f} | auroc {r['auroc']:.3f}")
        accuracy_table.append(acc_row)
        f1_table.append(f1_row)
        auroc_table.append(auroc_row)
        if i + 1 < len(families):
            next_accuracy.append(acc_row[i + 1])
            next_auroc.append(auroc_row[i + 1])

        seen_acc = sum(acc_row[:i + 1]) / (i + 1)
        seen_accuracy_per_task.append(seen_acc)
        log(f"  average accuracy on seen LLMs: {seen_acc:.3f} | forgetting so far: {forgetting(accuracy_table):.3f}"
            f" | task time {task_time:.0f}s | stored memory {stored_mb:.2f} MB")
        task_info.append({"task": i + 1, "family": family, "time": task_time, "stored_mb": stored_mb,
                          "stored_word_ids_mb": stored_ids_mb,
                          "train_texts": len(train_samples), **replay_info})

        # save a checkpoint (after every task for cgl, only at the end for the other methods to save space)
        if method == "cgl" or i == len(families) - 1:
            torch.save({"detector": detector.state_dict(),
                        "generator": generator.state_dict() if generator else None,
                        "words": tokenizer.words, "config": config, "families": families, "task": i + 1},
                       f"{folder}/checkpoints/after_task{i + 1}.pt")

    # final results 
    n = len(families)
    results = {
        "method": run_name,
        "seed": config["seed"],
        "families": families,
        "accuracy_table": accuracy_table,
        "f1_table": f1_table,
        "auroc_table": auroc_table,
        "final_accuracy": sum(accuracy_table[-1]) / n,
        "final_f1": sum(f1_table[-1]) / n,
        "final_auroc": sum(auroc_table[-1]) / n,
        # accuracy on each LLM straight after it arrived (how well the detector keeps up with new LLMs)
        "new_llm_accuracy": sum(accuracy_table[i][i] for i in range(n)) / n,
        "forgetting": forgetting(accuracy_table),
        "unseen_accuracy": unseen_accuracy(accuracy_table),
        "next_llm_accuracy": float(np.mean(next_accuracy)) if next_accuracy else float("nan"),
        "next_llm_auroc": float(np.mean(next_auroc)) if next_auroc else float("nan"),
        "train_time": sum(t["time"] for t in task_info),     # training only (testing time not counted)
        "stored_mb": task_info[-1]["stored_mb"],
        "stored_word_ids_mb": task_info[-1]["stored_word_ids_mb"],
        # lists with one value per task (to see how things change as more LLMs arrive)
        "seen_accuracy_per_task": seen_accuracy_per_task,
        "next_accuracy_per_task": next_accuracy,
        "next_auroc_per_task": next_auroc,
        "time_per_task": [t["time"] for t in task_info],
        "stored_mb_per_task": [t["stored_mb"] for t in task_info],
        "tasks": task_info,
    }
    save_json(results, folder + "/results.json")
    log("")
    log("#" * 60)
    log(f"FINAL ({run_name}): accuracy {results['final_accuracy']:.3f} | F1 {results['final_f1']:.3f} | "
        f"AUROC {results['final_auroc']:.3f} | forgetting {results['forgetting']:.3f} | "
        f"next-LLM AUROC {results['next_llm_auroc']:.3f} | time {results['train_time']:.0f}s | "
        f"memory {results['stored_mb']:.1f} MB")
    log("#" * 60)
    log_file.close()
    return results

# scores and results tables
# Metrics (accuracy, F1, AUROC, forgetting) and the results tables for the report.

def test_detector(detector, samples):
    probs, labels = [], []
    detector.eval()
    for ids, y, _, _ in make_batches(samples, 64, shuffle=False):
        probs += detector.predict(ids).tolist()
        labels += y.tolist()
    predictions = [1 if p > 0.5 else 0 for p in probs]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "auroc": roc_auc_score(labels, probs) if len(set(labels)) == 2 else float("nan"),
    }


def forgetting(table):
    # For every old family: (best accuracy it had AFTER it was learned) - (accuracy at the end).
 
    table = np.array(table)
    last = len(table) - 1
    if last == 0:
        return 0.0
    drops = [table[j:last, j].max() - table[last, j] for j in range(last)]
    return float(np.mean(drops))


def unseen_accuracy(table):
    #Average accuracy on families the detector has NOT trained on yet.
    table = np.array(table)
    values = [table[i, j] for i in range(len(table)) for j in range(i + 1, len(table[0]))]
    return float(np.mean(values)) if values else float("nan")


def average_results(runs):
    #Average the results of the same method over several seeds (and keep the spread).

    avg = {"method": runs[0]["method"], "families": runs[0]["families"], "n_seeds": len(runs)}
    for key, value in runs[0].items():
        if isinstance(value, (int, float)) and key != "seed":
            values = [r[key] for r in runs]
            avg[key] = float(np.mean(values))
            avg[key + "_std"] = float(np.std(values))
        elif isinstance(value, list) and value and not isinstance(value[0], (str, dict)):
            avg[key] = np.mean([r[key] for r in runs], axis=0).tolist()
    return avg


# results tables

def print_comparison(all_results, folder):
    #Print all the results as text tables
    n_tasks = len(all_results[0]["families"])
    n_seeds = all_results[0].get("n_seeds", 1)

    # table 1: how things change as new LLMs arrive (one block per part of the research question)
    blocks = [("seen_accuracy_per_task", "Accuracy on all LLMs seen so far (higher = less forgetting)", ".3f", 0),
              ("next_auroc_per_task", "AUROC on the NEXT, unseen LLM (0.5 = guessing)", ".3f", 1),
              ("time_per_task", "Training time for each new LLM (s)", ".0f", 0),
              ("stored_mb_per_task", "Memory kept between LLMs (MB)", ".1f", 0)]
    for key, title, fmt, first in blocks:
        print(f"\n{title}")
        print(f"{'after LLM':<14}" + "".join(f"{i:>7}" for i in range(1 + first, n_tasks + 1)))
        for r in all_results:
            print(f"{r['method']:<14}" + "".join(f"{v:>7{fmt}}" for v in r[key]))

    # table 2: accuracy on every LLM after every task, for each method
    for r in all_results:
        names = r["families"]
        print(f"\nAccuracy on each LLM after each task: {r['method']}  (above the diagonal = not learned yet)")
        print(f"{'':<20}" + "".join(f"{n[:7]:>8}" for n in names))
        for i, row in enumerate(r["accuracy_table"]):
            print(f"{'after ' + names[i]:<20}" + "".join(f"{v:>8.2f}" for v in row))

    # table 3: the final numbers (also saved for the report)
    def cell(r, key, fmt):
        text = format(r[key], fmt)
        if r.get(key + "_std", 0) > 0:
            text += " ± " + format(r[key + "_std"], fmt)
        return text

    lines = ["| method | final accuracy | new-LLM accuracy | final F1 | final AUROC | forgetting | next-LLM accuracy "
             "| next-LLM AUROC | train time (s) | memory kept (MB) | memory as word ids (MB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in all_results:
        lines.append(f"| {r['method']} | {cell(r, 'final_accuracy', '.3f')} | {cell(r, 'new_llm_accuracy', '.3f')} | "
                     f"{cell(r, 'final_f1', '.3f')} | "
                     f"{cell(r, 'final_auroc', '.3f')} | {cell(r, 'forgetting', '.3f')} | "
                     f"{cell(r, 'next_llm_accuracy', '.3f')} | {cell(r, 'next_llm_auroc', '.3f')} | "
                     f"{cell(r, 'train_time', '.0f')} | {cell(r, 'stored_mb', '.2f')} | {cell(r, 'stored_word_ids_mb', '.2f')} |")
    with open(folder + "/comparison_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nFinal results" + (f" (average of {n_seeds} seeds)" if n_seeds > 1 else "") + ":")
    print("\n".join(lines))


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
