# reward_utils.py


def compute_shaped_reward(result_after_action, player_name):


    reward = 0.0

    # Intermediate shaping
    if result_after_action.get("action_taken") == "pass":
        reward -= 0.01
    else:
        reward += 0.02

    # If player finished cards
    if result_after_action.get("round_over"):
        if result_after_action.get("player") == player_name:
            reward += 0.1

    return reward


def compute_terminal_reward(final_scores, player_name):

    sorted_players = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    ranking = [p[0] for p in sorted_players]

    if ranking[0] == player_name:
        return 3.0
    elif ranking[1] == player_name:
        return 2.0
    elif ranking[2] == player_name:
        return 1.0
    return 0.0