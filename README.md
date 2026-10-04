

## What is compared

Four ways of keeping a detector up to date are tested:

- **static:** learns the first AI tool only and never changes.
- **finetune:** learns only the newest AI tool each time.
- **full_retrain:** starts from zero and learns everything again each time.
- **cgl:** learns the newest AI tool plus fake reminders of the old ones. This is my method.

## The data

The project uses **MAGE**, a free public collection of texts written by people and by 27 different
AI tools.

## The files

- `main.py` – start here; it runs everything
- `config.yaml` – the settings, such as how much data to use and the list of AI tools
- `data.py` – loads the texts and splits them into the 11 steps
- `models.py` – the detector and the memory helper
- `train.py` – trains and tests the four approaches
- `requirements.txt` – the extra Python packages needed

## How to run it

**Step 1.** Install Python (version 3.10 to 3.12).

**Step 2.** Download MAGE's three files (`train.csv`, `valid.csv` and `test.csv`) and put them in
a folder called `data/mage` inside the project folder.

**Step 3.** Open the project folder in VS Code. Open a terminal (*Terminal → New Terminal*) and
type these lines one at a time:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Step 4.** Run the project. Type each command on one line and press Enter. Don't use VS Code's
Run button.

Check that the data loads properly (about one minute):

```
python main.py --mode check
```

Do a quick test to make sure everything works (a few minutes):

```
python main.py --mode compare --max_per_task 300 --epochs 1 --gen_steps 1000 --output outputs/test
```

Do the real run (about 30 to 40 minutes):

```
python main.py --mode compare --max_per_task 1000 --epochs 2 --gen_steps 3000 --output outputs/final
```

**Optional.** Try the detector on your own text:

```
python main.py --mode detect --checkpoint outputs/final/cgl/checkpoints/after_task11.pt --text "Paste some text here"
```

## What you see

All results are printed in the terminal. At the end of the real run you will see how each approach
did after every step, and a final table comparing the four. The numbers are also saved in the
`outputs/final` folder.

What the main numbers mean:

- **Accuracy:** how often the detector is right. 0.5 is the same as guessing; 1.0 is always right.
- **Forgetting:** how much worse it got on old AI tools after learning new ones. Lower is better.
- **Training time:** how many seconds it took to learn. Lower is better.
- **Memory kept:** how much storage it needs between steps. Lower is better.


