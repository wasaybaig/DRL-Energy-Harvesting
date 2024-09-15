# DRL-Energy-Harvesting

This repository contains the code for the paper titled **"Navigating Boundaries in Quantifying Robustness: A DRL Expedition for Non-Linear Energy Harvesting IoT Networks"** published in IEEE Communication Letters. The paper investigates deep reinforcement learning (DRL) approaches to optimize the data rate of energy-harvesting-enabled IoT devices in a cognitive radio-aided non-orthogonal multi-access (CR-NOMA) network. 

You can access the paper [here](https://ieeexplore.ieee.org/document/10659082).

## Overview

The primary objective of this work is to maximize the data rate of a resource-constrained secondary device (RCSD) coexisting with multiple primary devices (PDs) in a CR-NOMA network. This is done by optimizing the RCSD’s time-sharing coefficient and transmit power using convex optimization and DRL algorithms, while considering a realistic nonlinear energy harvesting (EH) model.

### Key Contributions

- **DRL Algorithms:** Compares the effectiveness of five DRL algorithms—DDPG, PER-DDPG, CER-DDPG, TD3, and PPO—for optimal control of time-sharing and power allocation parameters.
- **Throughput Maximization:** Formulates the throughput maximization problem for an RCSD operating in a CR-NOMA network using convex optimization and DRL algorithms.
- **Nonlinear EH Model:** Implements a realistic nonlinear EH model for accurate performance analysis.

