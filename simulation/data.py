from pathlib import Path
import pandas as pd

from simulation.utils import get_round


class Data:
    '''
    Loads pertinent data to forward model NCAA tournament
    based on a given probabilities from a submission file.
    Note that the submission file will be handled seperately.
    '''

    def __init__(self, mw=None, dir='./input'):
        if mw is None:
            raise ValueError('Tournament type not set')
        path = Path(dir)
        self.mw = mw.upper()
        self.seasons = pd.read_csv(path/(f'{self.mw}Seasons.csv'))
        self.teams = pd.read_csv(path/(f'{self.mw}Teams.csv'))
        if mw == 'W':
            self.slots = [
                pd.read_csv(path/(f'WNCAATourneySlots.csv')),
            ]
        else:
            self.slots = pd.read_csv(path/(f'MNCAATourneySlots.csv'))
        new_seeds = pd.read_csv(path/"2024_tourney_seeds.csv")
        new_seeds['Season'] = 2024
        new_seeds = new_seeds.drop(columns='Tournament')
        self.seeds = pd.concat([
            pd.read_csv(path/f"{self.mw}NCAATourneySeeds.csv"),
            new_seeds
        ], ignore_index=True)
        self.seedyear_dict, self.seedyear_dict_rev = self.build_seed_dicts()
        self.t_dict = (self.teams.set_index('TeamID')['TeamName'].to_dict())
        self.t_dict_rev = {v: k for k, v in self.t_dict.items()}

    def build_seed_dicts(self):
        seedyear_dict = {}
        seedyear_dict_rev = {}

        for s in self.seeds['Season'].unique():
            seed_data = self.seeds.query('Season == @s')
            s_dict = (seed_data.set_index('Seed')['TeamID']
                               .to_dict())
            s_dict_rev = {v: k for k, v in s_dict.items()}
            seedyear_dict.update({s: s_dict})
            seedyear_dict_rev.update({s: s_dict_rev})
        return seedyear_dict, seedyear_dict_rev

    def get_round(self, season, t1_id, t2_id):
        return get_round(
            season,
            t1_id,
            t2_id,
            self.seedyear_dict_rev
         )