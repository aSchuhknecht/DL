import torch
from pickle import load
from torch.utils.data import Dataset, DataLoader, random_split
from tournament.utils import load_recording
from .player import extract_featuresV2

DATASET_PATH = "train_data"

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path=DATASET_PATH):
        from glob import glob
        from os import path
        self.data = []
        for team in ["team1", "team2"]:
            for f in glob(path.join(dataset_path, team, '*.pkl')):
                game_data = load_recording(f)
                for frame in game_data:
                    # team state is a list of 2 dicts, so we can get actions for 2 jurgen agents in one go.
                    jurgen1, jurgen2 = frame['team1_state'] if team == 'team1' else frame['team2_state']
                    soccer_state = frame['soccer_state']
                    opponent_state = frame['team2_state'] if team == 'team1' else frame['team1_state']
                    jurgen1_id = jurgen1['kart']['player_id'] % 2
                    jurgen2_id = jurgen2['kart']['player_id'] % 2

                    features1 = extract_featuresV2(jurgen1, soccer_state, opponent_state, jurgen1_id)
                    features2 = extract_featuresV2(jurgen2, soccer_state, opponent_state, jurgen2_id)

                    all_actions = frame['actions']

                    # Use idx 0, 2 for team 1 games and 1, 3 for team 2 games
                    offset = team == 'team2'

                    actions1 = [all_actions[0 + offset]['acceleration'], all_actions[0 + offset]['steer'], all_actions[0 + offset]['brake']]
                    actions2 = [all_actions[2 + offset]['acceleration'], all_actions[2 + offset]['steer'], all_actions[2 + offset]['brake']]

                    self.data.append((features1, torch.tensor(actions1)))
                    self.data.append((features2, torch.tensor(actions2)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path=DATASET_PATH, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)
    train, test = random_split(loader.dataset, [0.8, 0.2])
    train_loader = DataLoader(train, batch_size=loader.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=loader.batch_size, shuffle=False)

    return train_loader, test_loader