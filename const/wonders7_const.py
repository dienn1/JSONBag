from typing import Literal

features = [
    'Round', 'Tick', 'ActionsCount(BuildFromDiscard)',
    'ActionsCount(DiscardCard)', 'ActionsCount(PlayCard)',
    'ActionsCount(BuildStage)', 'CardTypeCount(CivilianStructures)',
    'CardTypeCount(ManufacturedGoods)', 'CardTypeCount(MilitaryStructures)',
    'CardTypeCount(RawMaterials)', 'CardTypeCount(Guilds)',
    'CardTypeCount(CommercialStructures)',
    'CardTypeCount(ScientificStructures)', 'Resources(Clay)',
    'Resources(Wood_Ore)', 'Resources(Ore)', 'Resources(Textile)',
    'Resources(Wood)', 'Resources(Wood_Clay)', 'Resources(Wood_Stone)',
    'Resources(ScienceWild)', 'Resources(Shield)', 'Resources(Glass)',
    'Resources(Ore_Clay)', 'Resources(Stone)', 'Resources(Stone_Clay)',
    'Resources(Papyrus)', 'Resources(Compass)', 'Resources(RareWild)',
    'Resources(Coin)', 'Resources(BasicWild)', 'Resources(Stone_Ore)',
    'Resources(Cog)', 'Resources(Tablet)', 'Resources(Victory)'
]

game_name = "Wonders7"
actual_name = "7 Wonders"
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
token_filters = [None, None, ["wonderBoard", "seed"]]
tokenize_mode: Literal["both", "ordered", "unordered", "char"] = "unordered"

agent_paths = list(f"{game_name}/agents/{agent_types[i]}/" for i in range(len(agent_types)))
param_paths = list(f"{game_name}/gameParams/param{i}/" for i in range(1, param_count+1))
seed_paths = list(f"{game_name}/seeds/seed{i}/" for i in range(1, seed_count+1))

agent_dataset = [agent_paths, agent_labels]
param_dataset = [param_paths, list(f"param{i}" for i in range(1, param_count+1))]
seed_dataset = [seed_paths, list(f"seed{i}" for i in range(1, seed_count+1))]
