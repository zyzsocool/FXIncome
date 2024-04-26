import random


def simulate_games(num_games):
    scores = {"A": 0, "B": 0}
    outcomes = {"A Wins": 0, "B Wins": 0, "Tie": 0}

    for _ in range(num_games):
        # A choose Scissors with possibility 1/3, Rock with possibility 2/3
        a_move = random.choices(["Scissors", "Rock"], [1 / 3, 2 / 3], k=1)[0]
        # B choose Paper with possibility 1/3, Rock with possibility 2/3
        # b_move = random.choices(["Paper", "Rock"], [1 / 3, 2 / 3], k=1)[0]
        b_move = random.choices(["Paper", "Rock"], [1 / 3, 2 / 3], k=1)[0]

        if a_move == b_move:
            outcomes["Tie"] += 1
        elif (a_move == "Rock" and b_move == "Scissors") or (
            a_move == "Scissors" and b_move == "Paper"
        ):
            scores["A"] += 1
            scores["B"] -= 1
            outcomes["A Wins"] += 1
        else:
            scores["A"] -= 1
            scores["B"] += 1
            outcomes["B Wins"] += 1

    for player, score in scores.items():
        print(f"{player}'s score: {score / num_games * 100}%")

    for outcome, count in outcomes.items():
        print(f"{outcome}: {count / num_games * 100}%")


simulate_games(1000000)
