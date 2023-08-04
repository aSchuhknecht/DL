import subprocess
import random
import re
import os

how_many = 10
opponents = ["jurgen", "yann", "geoffrey", "yoshua"]
# goals_needed = {
#     opp: 1 if opp in ["yann", "jurgen", "image_jurgen"] else 2 for opp in opponents
# }
goals_needed = 1

# setup directories
if not os.path.exists("train_data/team1"):
    os.makedirs("train_data/team1")

if not os.path.exists("train_data/team2"):
    os.makedirs("train_data/team2")

for opp in opponents:
    wins = 0
    i = 0
    while wins < how_many:
        out_file = f"train_data/team1/jurgen_{opp}.{i}.pkl"
        command = [
            "python",
            "-m",
            "tournament.runner",
            "jurgen_agent",
            f"{opp}_agent" if opp != "ai" else "AI",
            "-s",
            out_file,
            "--ball_location",
            str(random.uniform(-1, 1)),
            str(random.uniform(-1, 1))
        ]

        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        match_score = result.stdout.strip()
        pattern = r'\[(\d, \d)\]'
        score = re.search(pattern, match_score).group(1).split(",")
        score1, score2 = int(score[0].strip()), int(score[1].strip())

        if score1 < goals_needed:
            os.remove(out_file)
        else:
            wins += 1
        print(f"Game: {out_file}\njurgen {score1} - {opp} {score2}")
        i += 1

    i = 0
    wins = 0
    while wins < how_many:
        out_file = f"train_data/team2/{opp}_jurgen.{i}.pkl"
        command = [
            "python",
            "-m",
            "tournament.runner",
            f"{opp}_agent" if opp != "ai" else "AI",
            "jurgen_agent",
            "-s",
            out_file,
            "--ball_location",
            str(random.uniform(-1, 1)),
            str(random.uniform(-1, 1))
        ]

        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        match_score = result.stdout.strip()
        pattern = r'\[(\d, \d)\]'
        score = re.search(pattern, match_score).group(1).split(",")
        score1, score2 = int(score[0].strip()), int(score[1].strip())

        if score2 < goals_needed:
            os.remove(out_file)
        else:
            wins += 1
        print(f"Game: {out_file}\njurgen {score2} - {opp} {score1}")
        i += 1