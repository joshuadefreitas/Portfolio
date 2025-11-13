# Reinforcement Learning Agent for Futures Trading

This project builds a **reinforcement learning (RL)** framework for trading futures in a simulated market.

## 🎯 Goal

Design an RL agent that learns to:
- Open, hold, and close positions
- Manage risk and drawdowns
- Improve risk-adjusted returns vs. simple benchmarks

## 🧱 Planned Components

- Custom Gym-style trading environment
- State space with returns, volatility, and microstructure features
- PPO / SAC-based agents
- Reward functions including:
  - Sharpe-like metrics
  - Drawdown penalties
  - Transaction cost / slippage
- Evaluation against rule-based strategies

## 🛠 Tech Stack

- Python
- PyTorch or Stable-Baselines3
- Gymnasium / custom environment
- pandas, NumPy

This project connects **RL theory** with **practical futures trading logic**.
