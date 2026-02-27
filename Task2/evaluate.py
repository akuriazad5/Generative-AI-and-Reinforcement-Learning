# evaluate.py


import asyncio
from chefshatgym.src.rooms.room import Room
from chefshatgym.src.agents.random_agent import RandomAgent
from agent import SparseRewardDQN
from reward_utils import compute_terminal_reward


async def evaluate():

    games = 50

    wins = 0
    total_score = 0
    total_sparse_reward = 0

    for i in range(games):

        room = Room(
            run_remote_room=False,
            room_name=f"EvalRoom_{i}",
            max_matches=10,
            save_logs_game=False,
            save_game_dataset=False,
        )

        agent_rl = SparseRewardDQN(
            name="RL_Agent",
            train=False,
            epsilon=0.0,
        )

        room.connect_player(agent_rl)
        room.connect_player(RandomAgent(name="Random1"))
        room.connect_player(RandomAgent(name="Random2"))
        room.connect_player(RandomAgent(name="Random3"))

        await room.run()

        scores = room.final_scores

        # Score accumulation
        total_score += scores["RL_Agent"]

        # Sparse reward accumulation
        sparse_reward = compute_terminal_reward(scores, "RL_Agent")
        total_sparse_reward += sparse_reward

        # Win check
        if max(scores, key=scores.get) == "RL_Agent":
            wins += 1

    print("\n========== EVALUATION RESULTS ==========\n")
    print(f"Win Rate: {wins/games:.2f}")
    print(f"Average Score per Game: {total_score/games:.2f}")
    print(f"Average Sparse Reward: {total_sparse_reward/games:.2f}")
    print("----------------------------------------\n")


if __name__ == "__main__":
    asyncio.run(evaluate())