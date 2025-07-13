from typing import Literal

features = [
    'Tick',
    # "FinalScore(Player-0)"
    'ColumnPieceCount(0)',
    'ColumnPieceCount(1)',
    'ColumnPieceCount(2)',
    'ColumnPieceCount(3)',
    'ColumnPieceCount(4)',
    'ColumnPieceCount(5)',
    'ColumnPieceCount(6)',
    'ColumnPieceCount(7)',
    'ColumnPieceCount(8)',
    'ColumnPieceCount(9)',
    'ColumnPieceCount(10)',
    'ColumnPieceCount(11)',
]

game_name = "Connect4"
actual_name = game_name
no_score = True
n_players = 2
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
token_filters = [None, ["width", "height"]]
tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "ordered"

agent_paths = list(f"{game_name}/agents/{agent_types[i]}/" for i in range(len(agent_types)))
param_paths = list(f"{game_name}/gameParams/param{i}/" for i in range(1, param_count + 1))

agent_dataset = [agent_paths, agent_labels]
param_dataset = [param_paths, list(f"param{i}" for i in range(1, param_count + 1))]
