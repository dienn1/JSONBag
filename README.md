# JSONBag: A generic game trajectory representation

## The Paper

## Overview
Run **analysis_main.py** for the main results with PNNS using JSD.
<br>Uncomment the game you want to analyze (the script will only run with the first game in the list); change ``n`` to the number of runs (different train/test splits)
<br><br><img src="./misc/analysis-main-games.png?raw=true">

Run **any_model_test.py** to test JSON-Bag with any other machine learning model. Here, the JSON-Bags are vectorized. Choose the game same as above. Choose model the same way; any classification model can be plugged in, assuming the same API as sklearn.
<br><br><img src="./misc/any_model_test_param.png?raw=true">

## JSON-Bag Tokenizer
The tokenizer is the function ``tokenize(...)``  in **tokenizer.py** that takes in a JSON object (loaded as ``dict``), the tokenization mode (``ordered, unordered, both, char``, more details in the paper), and the option for binning/pairing x, y coordinates in grid-based games, and **output a list of tokens**.

| Game                                                                                    | Tokenization mode                                                 |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| 7 Wonders<br>Dominion<br>Sea Salt and Paper<br>Can't Stop<br>Connect4<br>Dots and Boxes | unordered<br>unordered<br>unordered<br>both<br>ordered<br>ordered |

TODO: Implement the tokenizer as a class with more robust customization

## Handcrafted features baseline
### Universal features
All games have these features:
- Game tick (engine specifics)
- Turn count
- each player's score at game over
- rate of change and intercept of each player's scores throughout the game
	- Scores of a player are recorded periodically throughout a game trajectory into a score vector $\mathbf{s}$, a linear regression model is fitted to predict $s_i$: $w \times i + b = s_i$, where $i$ is the index of the score. We extract $w$ and $b$ for each player as features.

### Game-specific features
In the folder **const**, each game has a gamename_const.py with a list ``features`` containing game-specific handcrafted features.

## Game Data
The games used for the experiments are implemented in [TAG Framework](https://github.com/GAIGResearch/TabletopGames). The fork used by the experiments is [link]. All data is generated with this fork.

Due to size, the raw data is omitted here. In each game folder, there are some example raw game trajectories (concatenation of all JSON game states in a trajectory). The folders also include the full JSON-Bags of game trajectories
