# LSTPO  
### Preference-Guided Meta-Learning for Cross-Domain Time Series Forecasting

> **Learning when to remember far away, and when to listen to the recent past.**

LSTPO is a research codebase for **cross-domain time series forecasting** based on the paper **“Preference Guided Meta-Learning for Cross Domain Time Series Forecasting”**.  
Instead of treating all temporal patterns equally, LSTPO models forecasting as a **time-dependent preference learning problem**, where the model dynamically learns whether **long-term** or **short-term** temporal dependency is more useful across domains.

<p align="center">
  <img src="assets/teaser.png" alt="LSTPO teaser" width="85%">
</p>

---

## Why LSTPO?

Time series from different domains may look very different, but they often share deeper temporal structures.  
LSTPO is built on one simple idea:

- some domains rely more on **long-term dependencies** (trend, seasonality, cycles),
- others rely more on **short-term dependencies** (local fluctuation, abrupt changes),
- and robust cross-domain forecasting requires a model that can **adapt its temporal preference over time and across domains**.

To achieve this, LSTPO combines:

- **time series segmentation** into long- and short-term blocks,
- **temporal preference optimization** inspired by direct preference optimization,
- **meta-learning** to reduce catastrophic forgetting during multi-domain training,
- and **fast target-domain adaptation** for transfer and few-shot forecasting.  
  :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Highlights

- **Cross-domain forecasting from a preference perspective**  
  LSTPO reframes cross-domain forecasting as learning *which temporal dependency to trust*.

- **Dynamic long/short temporal preference modeling**  
  It explicitly compares long-term and short-term temporal blocks and optimizes their preference relationship during training. :contentReference[oaicite:4]{index=4}

- **Meta-learning for knowledge retention**  
  A dual-encoder design helps preserve domain-specific knowledge while accumulating cross-domain temporal preferences. :contentReference[oaicite:5]{index=5}

- **Strong few-shot transferability**  
  The paper reports strong gains under limited-data settings, showing that LSTPO is especially effective when adaptation data is scarce. :contentReference[oaicite:6]{index=6}

- **A practical research framework**  
  This repository is organized for reproducible experiments, ablations, evaluation, and preference visualization.

---

## Framework Overview

LSTPO follows a **three-stage pipeline**:

1. **Time Series Segmentation**  
   The input series is segmented into **long-term** and **short-term** blocks.  
   These blocks serve as candidates for temporal preference modeling.

2. **Meta-Learning-Based Preference Fine-Tuning**  
   A single-domain encoder learns domain-specific temporal preferences, while a cross-domain encoder aggregates them across domains to improve transferability and reduce forgetting.

3. **Adaptation and Forecasting**  
   The fine-tuned model rapidly adapts to a target domain and produces final forecasts.  
   This is the core workflow illustrated in the paper’s framework figure. :contentReference[oaicite:7]{index=7}

---

## Reported Results

According to the paper, LSTPO:

- achieves the **best result in 13 out of 18 evaluated cases**,
- reduces error by **11.9% in MSE** and **7.2% in MAE** relative to UniTime on the reported benchmark summary,
- and shows strong performance in **few-shot forecasting** with only **10% training data**.  
  :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

The paper evaluates on **nine real-world benchmark datasets** spanning multiple domains, and reports that LSTPO is particularly strong in robust cross-domain and few-shot settings. :contentReference[oaicite:10]{index=10}

---

## Repository Structure

```text
repo/
  README.md
  requirements.txt

  configs/
    base.yaml
    train.yaml
    eval.yaml
    ablation_tss.yaml

  src/
    __init__.py
    data/
      preprocessing.py
      datasets.py
      segmentation.py
      synthetic.py
    models/
      backbones/
        transformer_forecaster.py
        dlinear.py
      modules/
        preference.py
        adaptation.py
      lstpo.py
    losses/
      preference_losses.py
    training/
      callbacks.py
      loops.py
      trainer.py
    evaluation/
      metrics.py
      visualize.py
      evaluate.py
    utils/
      config.py
      io.py
      logging.py
      seed.py

  scripts/
    train.py
    evaluate.py
    run_synthetic.py
    visualize_preferences.py

  tests/
    test_dataset.py
    test_forward.py
