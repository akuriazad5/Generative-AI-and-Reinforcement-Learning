# train.py


import asyncio
import csv
import os
import matplotlib.pyplot as plt
from chefshatgym.src.rooms.room import Room
from chefshatgym.src.agents.random_agent import RandomAgent
from agent import SparseRewardDQN
from reward_utils import compute_terminal_reward


async def train():

    print("\n========== TRAINING STARTED ==========\n")

    TOTAL_MATCHES = 50
    csv_file = "training_metrics.csv"

    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Run", "FinalScore", "SparseReward"])

    room = Room(
        run_remote_room=False,
        room_name="SparseRewardRoom",
        max_matches=TOTAL_MATCHES,
        save_logs_game=False,
        save_game_dataset=False,
    )

    agent_rl = SparseRewardDQN(
        name="RL_Agent",
        train=True,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    )

    room.connect_player(agent_rl)
    room.connect_player(RandomAgent(name="Random1"))
    room.connect_player(RandomAgent(name="Random2"))
    room.connect_player(RandomAgent(name="Random3"))

    await room.run()

    print("\n========== TRAINING FINISHED ==========\n")

    final_scores = room.final_scores
    final_score = final_scores["RL_Agent"]
    sparse_reward = compute_terminal_reward(final_scores, "RL_Agent")

    print("Final Scores:", final_scores)
    print(f"Final Score: {final_score}")
    print(f"Sparse Reward: {sparse_reward}")

    with open(csv_file, mode="r") as file:
        row_count = sum(1 for _ in file)

    run_number = row_count  # header is row 1

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([run_number, final_score, sparse_reward])

    print("\nData appended to training_metrics.csv")

    runs = []
    final_scores_history = []

    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            runs.append(int(row["Run"]))
            final_scores_history.append(float(row["FinalScore"]))

    plt.figure()
    plt.plot(runs, final_scores_history)
    plt.xlabel("Training Run")
    plt.ylabel("Final Score (RL_Agent)")
    plt.title("Training Performance Across Runs")
    plt.savefig("learning_curve.png")
    plt.close()

    print("Plot updated: learning_curve.png\n")


if __name__ == "__main__":
    asyncio.run(train())