import os
import time

from natsort import os_sorted
from typing import *

from const import cantstop_const, dotsandboxes_const, wonders7_const, dominion_const, seasaltpaper_const, connect4_const
from utils import generate_no_history_states, load_states_index, generate_states_index, \
    generate_game_trajectories, generate_tokenized_dicts_from_trajectories, load_json_dir, generate_hidden_info_states

if __name__ == "__main__":
    # current_game = wonders7_const
    # current_game = seasaltpaper_const
    # current_game = dominion_const
    # current_game = connect4_const
    # current_game = dotsandboxes_const
    current_game = cantstop_const
    datasets = [
        current_game.agent_dataset,
        current_game.param_dataset,
        current_game.seed_dataset
    ]
    count = -1
    tokenize_modes: List[Literal["both", "ordered", "unordered", "char"]] = [
        "both",
        "ordered",
        "unordered",
        "char"
    ]
    t = time.time()
    for dataset_paths, dataset_labels in datasets:
        print(f"PREPROCESSING JSON {current_game.game_name}||{dataset_labels}")
        for path in dataset_paths:
            json_dir = path + "rawGameStates/"
            dir_scan = os.scandir(json_dir)
            json_entries = list(entry for entry in dir_scan if entry.is_file())
            json_entries = os_sorted(json_entries, key=lambda x: x.name)
            try:
                states_index, states_inverse_index = load_states_index(path)
            except FileNotFoundError:
                print("No states index file found, generating them...")
                states_index, states_inverse_index = generate_states_index(json_entries, path)

            no_history_path = path + "noHistory/"
            generate_no_history_states(json_entries, no_history_path, n=count)
            dir_scan = os.scandir(no_history_path + "json/")
            json_entries = list(entry for entry in dir_scan if entry.is_file())
            json_entries = os_sorted(json_entries, key=lambda x: x.name)

            # no_history_hidden_info_path = path + "noHistoryHiddenInfo/"
            # generate_hidden_info_states(json_entries, no_history_hidden_info_path, n=count)

            # for p in (no_history_path, no_history_hidden_info_path):
            for p in (no_history_path, ):
                generate_game_trajectories(states_inverse_index, current_game.game_name, p)
                trajectories, trajectory_names = load_json_dir(p + "trajectories/", name=p, return_obj=True)
                for mode in tokenize_modes:
                    generate_tokenized_dicts_from_trajectories(trajectories, trajectory_names, save_path=p, mode=mode)
            print("----------------------------------------------")
        print("----------------------------------------------*----------------------------------------------")
    print(f"FINISHED IN {time.time() - t} SECONDS")
