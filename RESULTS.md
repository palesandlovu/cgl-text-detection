# Results used in the paper (5 seeds: 42, 1, 2, 3, 4)

Produced with `python main.py --mode compare --seeds 42 1 2 3 4` (paper settings: 1,000 texts per task, 2 epochs, 3,000 generator steps) on a 2-core CPU without a GPU. Mean ± standard deviation over the five seeds. Times will differ on another computer.

| Method | Final acc | New-LLM acc | F1 | AUROC | Forgetting | Next AUROC | Time (s) | Memory (MB) |
|---|---|---|---|---|---|---|---|---|
| Static | 0.598 ± 0.001 | 0.598 ± 0.001 | 0.392 ± 0.027 | 0.607 ± 0.004 | 0.000 ± 0.000 | 0.577 ± 0.004 | 17 ± 0 | 0.00 ± 0.00 |
| Fine-tuning | 0.547 ± 0.026 | 0.717 ± 0.009 | 0.495 ± 0.074 | 0.559 ± 0.031 | 0.196 ± 0.033 | 0.556 ± 0.013 | 171 ± 3 | 0.00 ± 0.00 |
| Full retraining | 0.666 ± 0.012 | 0.667 ± 0.010 | 0.620 ± 0.061 | 0.738 ± 0.006 | 0.041 ± 0.013 | 0.603 ± 0.007 | 958 ± 31 | 14.68 ± 0.00 |
| Stored real texts (500) | 0.621 ± 0.015 | 0.707 ± 0.005 | 0.565 ± 0.047 | 0.670 ± 0.019 | 0.118 ± 0.010 | 0.589 ± 0.002 | 238 ± 4 | 0.10 ± 0.00 |
| CGL (diffusion replay) | 0.556 ± 0.025 | 0.673 ± 0.005 | 0.536 ± 0.039 | 0.606 ± 0.030 | 0.150 ± 0.032 | 0.517 ± 0.015 | 447 ± 8 | 1.87 ± 0.00 |
| CGL without replay | 0.483 ± 0.038 | 0.686 ± 0.007 | 0.449 ± 0.102 | 0.481 ± 0.017 | 0.240 ± 0.044 | 0.505 ± 0.019 | 443 ± 4 | 1.87 ± 0.00 |
| CGL with Gaussian generator | 0.556 ± 0.022 | 0.674 ± 0.005 | 0.530 ± 0.034 | 0.602 ± 0.022 | 0.150 ± 0.032 | 0.524 ± 0.013 | 190 ± 2 | 0.07 ± 0.00 |

Full retraining memory is raw text; as 16-bit word ids the same texts take 2.16 MB.

## Same-seed comparisons

| Comparison | Final acc | AUROC | Forgetting |
|---|---|---|---|
| CGL vs fine-tuning | +0.009 (CGL better in 3/5 seeds) | +0.047 (5/5) | −0.046 (5/5) |
| CGL vs CGL without replay | +0.073 (5/5) | +0.125 (5/5) | −0.090 (5/5) |
| CGL vs static | −0.042 (0/5) | −0.001 | +0.150 (0/5) |
| CGL vs Gaussian generator | 0.000 | +0.004 | 0.000 |
| Stored texts vs CGL | +0.065 (stored better in 5/5) | +0.065 (5/5) | −0.032 (4/5) |
