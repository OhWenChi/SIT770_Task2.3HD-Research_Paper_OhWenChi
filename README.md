## Results Summary

This repository contains the implementation and outputs for the research paper:

**Prompt-Induced Instability Undermines Reliable Bias Evaluation in Large Language Models**

The experiment evaluates prompt-induced fairness instability on the CrowS-Pairs benchmark using:

- `gpt2-medium`
- `Qwen/Qwen2.5-1.5B-Instruct`

Six semantically equivalent prompt variants were tested:

1. baseline
2. lexical paraphrasing
3. syntactic variation
4. politeness / hedging
5. repetition / emphasis
6. formatting variation

## Main Results

| Model | PIV | Mean Output Variance | Consistency Rate | Flip Rate | Naive PEBE Bias | Naive PEBE Variance Reduction | SA-PEBE Bias | SA-PEBE Variance Reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 Medium | 0.000107 | 0.002761 | 85.0% | 15.0% | 55.0% | 2.7% | 55.7% | -35.0% |
| Qwen2.5-1.5B-Instruct | 0.000036 | 0.008877 | 84.0% | 16.0% | 66.7% | 8.4% | 67.6% | -21.2% |

## Key Findings

The results show that prompt-induced fairness instability persists across both an older autoregressive model and a modern instruction-tuned model. GPT-2 Medium produced a 15.0% flip rate, while Qwen2.5-1.5B-Instruct produced a 16.0% flip rate. This means that around one in six examples changed stereotype preference solely due to semantically equivalent prompt reformulation.

Qwen2.5-Instruct achieved lower aggregate Prompt-Induced Variance (PIV), but higher mean output variance and a slightly higher flip rate. This suggests that instruction tuning may improve aggregate benchmark stability while still leaving local example-level instability unresolved.

Naive PEBE produced modest variance reduction for both models. However, Stability-Aware PEBE produced negative variance reduction, suggesting that calibration-based prompt weighting did not generalise reliably.

## Figures

### Prompt-Induced Flip Rate Across Models

![Prompt-Induced Flip Rate Across Models](results/figure_flip_rate_by_model.png)

### Prompt-Induced Variance Across Models

![Prompt-Induced Variance Across Models](results/figure_piv_by_model.png)

### Single-Prompt vs Naive PEBE vs Stability-Aware PEBE

![Single-Prompt vs PEBE](results/figure_pebe_comparison.png)

### Prompt-Induced Instability by Bias Type

![Bias Type Instability](results/figure_bias_type_instability_by_model.png)

## Reproducing the Experiment

Install dependencies:

```bash
pip install transformers accelerate datasets scipy matplotlib pandas numpy tqdm bitsandbytes
