from typing import Literal

features = [
    'Round', 'Tick',
    'ActionsCount(CrabDuo)',
    'ActionsCount(DrawAndDiscard)', 'ActionsCount(FishDuo)',
    'ActionsCount(SwimmerSharkDuo)', 'ActionsCount(Stop)',
    'ActionsCount(LastChance)', 'ActionsCount(BoatDuo)',
    'cardTypeCount(MULTIPLIER)', 'cardTypeCount(COLLECTOR)',
    'cardTypeCount(MERMAID)', 'cardTypeCount(DUO)',
    'cardSuiteCount(PENGUIN)', 'cardSuiteCount(SWIMMER)',
    'cardSuiteCount(SHELL)', 'cardSuiteCount(SHARK)',
    'cardSuiteCount(FISH)', 'cardSuiteCount(OCTOPUS)',
    'cardSuiteCount(SAILOR)', 'cardSuiteCount(MERMAID)',
    'cardSuiteCount(BOAT)', 'cardSuiteCount(CRAB)',
    'cardColorCount(LIGHT_ORANGE)', 'cardColorCount(WHITE)',
    'cardColorCount(PINK)', 'cardColorCount(LIGHT_BLUE)',
    'cardColorCount(BLUE)', 'cardColorCount(BLACK)',
    'cardColorCount(YELLOW)', 'cardColorCount(PURPLE)',
    'cardColorCount(GREEN)', 'cardColorCount(GREY)',
    'cardColorCount(ORANGE)'
]

game_name = "SeaSaltPaper"
actual_name = "Sea Salt and Paper"
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
tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "unordered"

agent_paths = list(f"{game_name}/agents/{agent_types[i]}/" for i in range(len(agent_types)))
param_paths = list(f"{game_name}/gameParams/param{i}/" for i in range(1, param_count+1))
seed_paths = list(f"{game_name}/seeds/seed{i}/" for i in range(1, seed_count+1))

agent_dataset = [agent_paths, agent_labels]
param_dataset = [param_paths, list(f"param{i}" for i in range(1, param_count+1))]
seed_dataset = [seed_paths, list(f"seed{i}" for i in range(1, seed_count+1))]
