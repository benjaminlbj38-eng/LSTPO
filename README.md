# LSTPO
### Preference-Guided Meta-Learning for Cross-Domain Time Series Forecasting

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg">
  <img alt="Task" src="https://img.shields.io/badge/Task-Time%20Series%20Forecasting-6a5acd.svg">
  <img alt="Setting" src="https://img.shields.io/badge/Setting-Cross--Domain-2e8b57.svg">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-orange.svg">
</p>

<p align="center">
  <b>Learning when to remember far away, and when to trust the recent past.</b>
</p>

LSTPO is a research codebase for **cross-domain time series forecasting** based on the paper **“Preference Guided Meta-Learning for Cross Domain Time Series Forecasting”**.  
Instead of treating all temporal patterns equally, LSTPO models forecasting as a **time-dependent preference learning problem**, where the model dynamically learns whether **long-term** or **short-term** temporal dependency is more useful across domains.

<p align="center">
  <img src="assets/framework.png" alt="LSTPO Framework" width="85%">
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

---

## Highlights

- **Cross-domain forecasting from a preference perspective**  
  LSTPO reframes cross-domain forecasting as learning *which temporal dependency to trust*.

- **Dynamic long/short temporal preference modeling**  
  It explicitly compares long-term and short-term temporal blocks and optimizes their preference relationship during training. :contentReference[oaicite:4]{index=4}
---

## 🧠 Framework Overview

LSTPO follows a three-stage pipeline:

1. **Time Series Segmentation**  
   The input sequence is decomposed into long-term and short-term temporal blocks.

2. **Preference-Guided Meta-Learning**  
   The model learns which temporal dependency is more informative across domains, while preserving domain-specific knowledge and mitigating catastrophic forgetting.

3. **Adaptation and Forecasting**  
   The learned preference is rapidly adapted to the target domain for final forecasting.

<p align="center">
  <img src="assets/framework.png" alt="LSTPO Framework" width="88%">
</p>
<p align="center">
  <em>Overview of the LSTPO framework.</em>
</p>

---

## Reported Results

According to the paper, LSTPO:

- achieves the **best result in 13 out of 18 evaluated cases**,
- reduces error by **11.9% in MSE** and **7.2% in MAE** relative to UniTime on the reported benchmark summary,
- and shows strong performance in **few-shot forecasting** with only **10% training data**.  
  :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

The paper evaluates on **nine real-world benchmark datasets** spanning multiple domains, and reports that LSTPO is particularly strong in robust cross-domain and few-shot settings. :contentReference[oaicite:10]{index=10}

---
## 🙏 Acknowledgements

This repository was developed with inspiration from and reference to the following excellent open-source projects:

- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [UniTime](https://github.com/liuxu77/UniTime)

We sincerely thank the authors and contributors of these projects for making their code publicly available and for advancing research in time series forecasting and cross-domain learning.

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
