# JSONBag: A generic game trajectory representation

## The Paper

## Overview
Run **analysis_main.py** for the main results with PNNS using JSD.
Uncomment the game you want to analyze (the script will only run with the first game in the list); change ``n`` to the number of runs (different train/test splits)
(image)

Run **any_model_test.py** to test JSON-Bag with any other machine learning model. Choose the game same as above. Choose model the same way; any classification model can be plugged in, assuming the same API as sklearn.
(image)

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
