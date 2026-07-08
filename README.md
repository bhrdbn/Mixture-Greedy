# Mixture-Greedy for Online Generative Model Selection

This repository contains the official implementation for:

**Mixture-Greedy for Online Generative Model Selection:  
Do We Always Need UCB in Diversity-Aware Multi-Armed Bandits?**

The code implements Mixture-Greedy, a simple online mixture-selection algorithm for choosing among multiple generative models under diversity-aware evaluation objectives. Unlike Mixture-UCB methods, Mixture-Greedy optimizes the empirical mixture objective directly, without adding an explicit UCB exploration bonus. Our experiments show that this simple strategy can converge faster and achieve strong performance across image, text, and text-to-image generation settings.

## Overview

In online generative model selection, we are given a fixed pool of pretrained generators. At each round, the algorithm chooses one generator, observes a generated sample, updates its empirical estimate of the objective, and adjusts the mixture weights over generators.

This repository supports experiments with:

- **Image-generation benchmarks**, including FFHQ, ImageNet, and LSUN-Bedroom.
- **Text-generation benchmarks**, such as city-name generation.
- **Text-to-image generation benchmarks**, including red-bird and dog-breed prompts.
- Multiple objectives, including:
  - Fréchet Distance / FD
  - Vendi Score
  - RKE / inverse-RKE
  - Kernel-based objectives

The main goal is to study whether diversity-aware mixture objectives can induce implicit exploration, making explicit UCB bonuses unnecessary in several practical settings.

## Method

Mixture-Greedy maintains a mixture distribution over a set of generators. At each round, it solves

```math
\alpha_t \in \arg\min_{\alpha \in \Delta_m} \widehat{L}_{t-1}(\alpha),
