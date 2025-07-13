from typing import Literal

features = [
    'Tick',
    'Turn',
    'ActionsCount(Pass)',
    'ActionsCount(AllocateDice)',
    'ActionsCount(RollDice)'
]

game_name = "CantStop"
actual_name = "Can't Stop"
n_players = 4
agent_types = [
    "mcts1",
    "mcts1-tuned",
    "mcts2-tuned",
    "osla",
    "random"
]
agent_labels = [
    "MCTS-V",
    "MCTS64",
    "MCTS128",
    "OSLA",
    "Random"
]
param_count = 4
seed_count = 4
tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "both"

agent_paths = list(f"{game_name}/agents/{agent_types[i]}/" for i in range(len(agent_types)))
param_paths = list(f"{game_name}/gameParams/param{i}/" for i in range(1, param_count + 1))
seed_paths = list(f"{game_name}/seeds/seed{i}/" for i in range(1, seed_count + 1))

agent_dataset = [agent_paths, agent_labels]
param_dataset = [param_paths, list(f"param{i}" for i in range(1, param_count + 1))]
seed_dataset = [seed_paths, list(f"seed{i}" for i in range(1, seed_count + 1))]
