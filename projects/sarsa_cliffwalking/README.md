# SARSA Cliff Walking (Reinforcement Learning)

## Overview

This project presents an implementation of the **SARSA (State–Action–Reward–State–Action)** algorithm using a Jupyter Notebook to solve the **Cliff Walking problem**. The agent learns to navigate a grid-based environment from a start state to a goal state while avoiding high-penalty cliff regions.

The focus of this project is on **on-policy learning**, where the agent updates its knowledge based on the actions it actually takes, resulting in safer and more stable navigation.

---

## Key Features

* Implementation of **SARSA (on-policy TD control)**
* Grid-based **Cliff Walking environment**
* **Epsilon-greedy policy** for exploration
* Step-by-step learning across episodes
* Jupyter Notebook for clear experimentation and visualization

---

## Algorithm

The SARSA update rule used:

[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma Q(s',a') - Q(s,a) \big]
]

* On-policy learning method
* Updates based on **next action selected by policy**
* Encourages **safe path selection**

---

## Implementation Details

* The environment is represented as a **grid world**
* The agent starts from an initial state and aims to reach the goal
* Falling into the cliff results in a **large negative reward**
* The model learns over multiple episodes to improve performance

---

## Tech Stack

* Python
* NumPy
* Jupyter Notebook

---

## Results

* The agent learns to **avoid cliff regions**
* Demonstrates **safe navigation behavior**
* Converges to a stable policy over time
* Highlights the trade-off between **risk and reward**

---

## File Structure

```
sarsa_cliffwalking/
│
├── SARSA_cliffwalking.ipynb
└── README.md
```

---

## How to Run

1. Open the notebook:

   ```
   SARSA_cliffwalking.ipynb
   ```
2. Run all cells sequentially
3. Observe the agent’s learning behavior

---

## Applications

* Reinforcement Learning fundamentals
* Safe decision-making systems
* Academic labs and coursework
* Machine Learning portfolio projects

---

## Conclusion

This project demonstrates how SARSA enables agents to learn **policy-aware and risk-sensitive strategies**. It is particularly useful in environments where **safety is more important than purely optimal performance**.

---

## Author

**Ayush Kumar Agarwal**
Reinforcement Learning | Problem Solving | Software Development

