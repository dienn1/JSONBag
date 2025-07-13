import random

if __name__ == '__main__':
    # values = [
    #     [3, 5, 7, 10],
    #     [1, 3, 5, 6, 10],
    #     [5, 10, 15, 20],
    #     [5, 10, 15, 20],
    #     [3, 5, 7, 10, 15],
    #     [1,3,5,7,10],
    #     [10,20,32,40,50],
    #     [10,20,30,40,50],
    #     [10,20,30,40,50]
    # ]
    # values = [
    #     [1, 2, 3],
    #     [0, 1, 2],
    #     [2, 3, 4],
    #     [2, 3, 4]
    # ]

    # values = [
    #     [2, 3, 4, 5],
    #     [1, 2, 3, 4],
    #     [3, 5, 7, 10],
    #     [0, 1, 2, 3],
    #     [2, 3, 4, 5],
    #     [0, 1, 2, 3],
    #     [2, 3, 4, 5],
    #     [0, 1, 2, 3],
    #     [2, 3, 4, 5],
    #     [0, 1, 2, 3],
    #     [2, 3, 4, 5],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4]
    # ]
    #
    # names = [
    #     "numberOfCardsDraw",
    #     "numberOfCardsDiscard",
    #     "roundStopCondition",
    #     "SHELL_COLLECTOR_BASE",
    #     "SHELL_COLLECTOR_INC",
    #     "OCTOPUS_COLLECTOR_BASE",
    #     "OCTOPUS_COLLECTOR_INC",
    #     "PENGUIN_COLLECTOR_BASE",
    #     "PENGUIN_COLLECTOR_INC",
    #     "SAILOR_COLLECTOR_BASE",
    #     "SAILOR_COLLECTOR_INC",
    #     "BOAT_DUO_BONUS",
    #     "FISH_DUO_BONUS",
    #     "CRAB_DUO_BONUS",
    #     "SWIMMER_SHARK_DUO_BONUS",
    #     "BOAT_MULTIPLIER",
    #     "FISH_MULTIPLIER",
    #     "CRAB_MULTIPLIER",
    #     "PENGUIN_MULTIPLIER",
    #     "SAILOR_MULTIPLIER"
    # ]

    # values = [
    #     [6, 8, 10, 12],
    #     [3, 4, 5, 6]
    # ]
    # names = [
    #     "gridSize",
    #     "winCount"
    # ]

    values = [
        [5, 7, 9, 11, 13],
        [5, 7, 9, 11, 13]
    ]
    names = [
        "gridWidth",
        "gridHeight"
    ]

    # values = [
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12],
    #     [2, 4, 6, 8, 10, 12]
    # ]
    # names = [
    #     "TWO_MAX",
    #     "THREE_MAX",
    #     "FOUR_MAX",
    #     "FIVE_MAX",
    #     "SIX_MAX",
    #     "SEVEN_MAX",
    #     "EIGHT_MAX",
    #     "NINE_MAX",
    #     "TEN_MAX",
    #     "ELEVEN_MAX",
    #     "TWELVE_MAX"
    # ]

    for n, v in zip(names, values):
        print(f"\"{n}\"" + ": " + str(random.choice(v)) + ",")
