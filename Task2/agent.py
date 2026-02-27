# agent.py
from chefshatgym.src.agents.agent_dqn import DQNAgent
from reward_utils import compute_shaped_reward, compute_terminal_reward


class SparseRewardDQN(DQNAgent):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def update_player_action(self, payload):

        # Only learn from this agent's actions
        if payload.get("player") != self.name:
            return

        if "observation_before" not in payload:
            return

        if not self.train:
            return

        shaped_reward = compute_shaped_reward(payload, self.name)


        payload["reward"] = shaped_reward

        super().update_player_action(payload)

    def update_match_over(self, payload):

        if not self.train:
            return

        terminal_reward = compute_terminal_reward(payload["scores"], self.name)

        if len(self.memory) > 0:

            last = self.memory[-1]

            self.memory[-1] = (
                last[0],                     # state
                last[1],                     # action
                last[2] + terminal_reward,   # updated reward
                last[3],                     # next_state
                last[4],                     # possible_actions
                last[5],                     # next_possible_actions
                True                         # done
            )

        # Let original replay handle batching
        if len(self.memory) > self.batch_size:
            super().replay()