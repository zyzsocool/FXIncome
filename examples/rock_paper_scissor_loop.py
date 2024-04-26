import numpy as np
import random
import itertools


def simulate_games(num_games, a_s, a_r, b_p, b_r, b_s):
    scores = {"A": 0, "B": 0}

    for _ in range(num_games):
        a_move = random.choices(["Scissors", "Rock"], [a_s, a_r], k=1)[0]
        b_move = random.choices(["Scissors", "Rock", "Paper"], [b_s, b_r, b_p], k=1)[0]

        if a_move == b_move:
            continue
        elif (a_move == "Rock" and b_move == "Scissors") or (
            a_move == "Scissors" and b_move == "Paper"
        ):
            scores["A"] += 1
            scores["B"] -= 1
        else:
            scores["A"] -= 1
            scores["B"] += 1

    return scores


def main():
    min_max_score = float("inf")
    best_a_s = best_a_r = 0
    best_b_s = best_b_r = best_b_p = 0

    for a_s in np.arange(0, 1.01, 0.01):
        a_r = 1 - a_s
        max_b_score = float("-inf")
        for b_p in np.arange(0, 1.01, 0.01):
            for b_r in np.arange(0, 1.01 - b_p, 0.01):
                b_s = 1 - b_p - b_r
                scores = simulate_games(100, a_s, a_r, b_p, b_r, b_s)
                if scores["B"] > max_b_score:
                    max_b_score = scores["B"]
                    best_b_s = b_s
                    best_b_r = b_r
                    best_b_p = b_p
        if max_b_score < min_max_score:
            min_max_score = max_b_score
            best_a_s = a_s
            best_a_r = a_r

    print(f"Best strategy for A: Scissors = {best_a_s}, Rock = {best_a_r}")
    print(
        f"Best strategy for B: Scissors = {best_b_s}, Rock = {best_b_r}, Paper = {best_b_p}, score = {min_max_score}"
    )


if __name__ == "__main__":
    main()
