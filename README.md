##PK Ndlovu

GitHub: https://github.com/palesandlovu/cgl-text-detection

## What is compared

Seven ways of keeping a detector up to date are tested:

- **static:** learns the first AI tool only and never changes.
- **finetune:** learns only the newest AI tool each time.
- **full_retrain:** starts from zero and learns everything again each time.
- **cgl:** learns the newest AI tool plus fake reminders of the old ones. This is my method.
- **cgl_no_replay:** cgl with the fake reminders switched off (slow encoder only). Shows what the replay adds.
- **cgl_gaussian:** cgl with the simplest possible generator instead of the diffusion model. Shows whether diffusion is needed.
- **exemplar:** keeps 500 real old texts (as word numbers) and mixes them into training. The simplest alternative to fake reminders.

## The data

The project uses **MAGE**, a free public collection of texts written by people and by 27 different
AI tools.

## The files

- `main.py` – start here; it runs everything
- `config.yaml` – the settings, such as how much data to use and the list of AI tools
- `data.py` – loads the texts and splits them into the 11 steps
- `models.py` – the detector and the memory helpers (diffusion and Gaussian generators, EWC)
- `train.py` – trains and tests all the methods
- `requirements.txt` – the extra Python packages needed

## How to run it

Download MAGE's three files (`train.csv`, `valid.csv` and `test.csv`) and put them in
a folder called `data/mage` inside the project folder.

Open a terminal (*Terminal → New Terminal*) and type these lines one at a time:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Run the project. Type each command on one line and press Enter. Don't use VS Code's
Run button.

Check that the data loads properly (about one minute):

```
python main.py --mode check
```

Do a quick test to make sure everything works (a few minutes):

```
python main.py --mode compare --max_per_task 300 --epochs 1 --gen_steps 1000 --output outputs/test
```

Do the real run, the one used in the paper (all seven methods, five seeds). The default settings in
`config.yaml` are the paper's settings (1,000 texts per AI tool, 2 epochs, 3,000 generator steps). This
takes several hours on a laptop CPU:

```
python main.py --mode compare --seeds 42 1 2 3 4 --output outputs/final
```

To run only some methods, add for example `--methods cgl finetune`. To run one method on its own:

```
python main.py --mode train --method cgl --output outputs/cgl_only
```

You can change any setting in config.yaml, or on the command line, for example --epochs 3, --lr 0.001,
--replay_ratio 0, --generator gaussian or --buffer_size 1000.

While it trains it prints the loss, accuracy, speed, RAM and how well the replay is working. After every
LLM it shows the accuracy on the old, new and unseen LLMs. Everything is also saved in the outputs folder
(train_log.txt, results.json and comparison_table.md).

## What you see

All results are printed in the terminal. At the end of the real run you will see how each approach
did after every step, and a final table comparing all seven (mean ± spread over the five seeds). The
numbers are also saved in `outputs/final/comparison_table.md`.

- **Accuracy:** how often the detector is right. 0.5 is the same as guessing; 1.0 is always right.
- **Forgetting:** how much worse it got on old AI tools after learning new ones (best accuracy after
  learning a tool, minus its accuracy at the end). Lower is better.
- **Training time:** how many seconds it took to learn. Lower is better.
- **Memory kept:** how much has to be stored between steps. Lower is better. For full retraining this is
  shown twice: as raw text, and as the word numbers the model actually reads.

## Try the detector on your own text

```
python main.py --mode detect --checkpoint outputs/final/seed42/cgl/checkpoints/after_task11.pt --text "Paste some text here"
```

It prints the chance that the text is AI-generated and its verdict. (With a single seed the checkpoint
is in `outputs/final/cgl/checkpoints/` instead.)

## See what the generator learned

```
python main.py --mode generate --checkpoint outputs/final/seed42/cgl/checkpoints/after_task11.pt --n 20
```

This only works for cgl checkpoints, because only cgl trains a generator.
