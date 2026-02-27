# Dataset Link - https://huggingface.co/datasets/cardiffnlp/tweet_eval



# Chef's Hat Reinforcement Learning -- Sparse / Delayed Reward Variant

## Assigned Variant

This project implements a **Sparse / Delayed Reward Reinforcement
Learning (RL) variant** for the Chef's Hat multi-agent card game.

Instead of giving large rewards at every step, the agent:

-   Receives **very small intermediate shaped rewards**
-   Receives a **large delayed terminal reward** based on final ranking:
    -   1st place → +3\
    -   2nd place → +2\
    -   3rd place → +1\
    -   4th place → 0

This encourages the agent to learn **long-term strategies** rather than
optimising short-term actions.

The RL agent is a **DQN-based agent** extended as `SparseRewardDQN`,
which: - Uses experience replay - Applies minimal action-level shaping -
Injects the terminal reward into the final transition of the episode

------------------------------------------------------------------------

## How to Run the Code

### 1. Training

Run training (default mode):

``` bash
python main.py
```

or

``` bash
python main.py --mode train
```

Training configuration:

-   Matches per training run: **50**
-   Players:
    -   1 RL agent
    -   3 Random agents
-   After each run:
    -   Final score is recorded
    -   Sparse terminal reward is computed
    -   Data is appended to `training_metrics.csv`
    -   `learning_curve.png` is updated

------------------------------------------------------------------------

### 2. Evaluation

To evaluate the trained agent:

``` bash
python main.py --mode eval
```

Evaluation setup:

-   50 evaluation games
-   Each game contains 10 matches
-   Opponents: 3 random agents
-   Reported metrics:
    -   Win rate
    -   Average score per game
    -   Average sparse reward

Exploration is disabled during evaluation (`epsilon = 0.0`) to measure
the learned policy.

------------------------------------------------------------------------

## Experiments Conducted

### Experiment 1 -- Sparse + Delayed Reward Training

The agent was trained using:

-   Small shaped rewards:
    -   Pass action → −0.01\
    -   Play action → +0.02\
    -   Finishing a round → +0.1\
-   Large terminal ranking reward:
    -   1st → +3, 2nd → +2, 3rd → +1, 4th → 0

Purpose: - Encourage long-term planning - Reduce reliance on immediate
rewards - Improve final ranking performance

------------------------------------------------------------------------

### Experiment 2 -- Training Performance Tracking

After each training run:

-   Final RL agent score is stored in `training_metrics.csv`
-   A learning curve (`learning_curve.png`) is generated:
    -   X-axis → Training run number\
    -   Y-axis → Final score

This is used to observe whether the agent improves over time.

------------------------------------------------------------------------

### Experiment 3 -- Policy Evaluation

The trained agent was evaluated against **3 random agents** with:

-   No exploration (`epsilon = 0`)
-   Metrics:
    -   Win rate
    -   Average score
    -   Average sparse reward

This measures the **true performance** of the learned policy.

------------------------------------------------------------------------

## How to Interpret the Results

### 1. Training Metrics (`training_metrics.csv`)

Columns:

-   `Run` → Training iteration number\
-   `FinalScore` → RL agent's final score\
-   `SparseReward` → Terminal ranking reward

Interpretation:

-   Increasing **FinalScore trend** → Agent is learning
-   More frequent **SparseReward values of 2 or 3** → Higher final
    rankings

------------------------------------------------------------------------

### 2. Learning Curve (`learning_curve.png`)

-   Upward trend → Performance improvement
-   Flat curve → No learning (reward too sparse or insufficient
    training)
-   High variance → Unstable strategy

Because rewards are delayed, learning progress may appear **slow in
early runs**.

------------------------------------------------------------------------

### 3. Evaluation Metrics

Key indicators:

-   **Win Rate**
    -   Main success metric

    -   0.25 indicates performance better than random (4 players)
-   **Average Score**
    -   Measures consistency across games
-   **Average Sparse Reward**
    -   Reflects average final ranking

------------------------------------------------------------------------

## Expected Learning Behaviour

Due to sparse and delayed rewards:

-   Early training may appear random
-   Improvement occurs gradually
-   Terminal reward dominates learning
-   Agent focuses on **match-level strategy** instead of single moves

------------------------------------------------------------------------

## Summary

This project demonstrates:

-   Sparse and delayed reward learning
-   Terminal ranking-based optimisation
-   Multi-agent RL training against random opponents
-   Performance tracking using CSV logs and learning curves
-   Policy evaluation using win rate and average score

The agent is expected to:

-   Learn slower than dense-reward agents
-   Develop stronger long-term strategies
-   Achieve improved final rankings over multiple training runs
