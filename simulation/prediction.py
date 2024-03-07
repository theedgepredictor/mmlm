import numpy as np

from simulation.utils import get_alt_game_id


class Prediction:
    '''
    Holds the game prediction and methods to query the probability
    or loss for each team. Can also call a winner based on who is
    favored or a random sample using the odds.
    '''

    def __init__(self, season, t1_id, t2_id, pred, t_dict, s_dict):

        self.t_dict = t_dict
        self.s_dict = s_dict
        self.game_id = f'{season}_{t1_id}_{t2_id}'
        self.season = season
        self.t1_id = t1_id
        self.t2_id = t2_id
        self.pred = pred

    def __repr__(self):
        if self.proba[self.t1_id] > .5:
            proba = self.proba[self.t1_id]
            win_name = self.t1_name
            lose_name = self.t2_name
        else:
            proba = self.proba[self.t2_id]
            win_name = self.t2_name
            lose_name = self.t1_name

        return (f'{proba:.1%} chance of '
                f'{win_name} beating {lose_name}')

    @property
    def t1_name(self):
        return self.t_dict[self.t1_id]

    @property
    def t2_name(self):
        return self.t_dict[self.t2_id]

    @property
    def alt_game_id(self):
        return get_alt_game_id(self.game_id)

    @property
    def proba(self):
        return {
            self.t1_id: self.pred,
            self.t2_id: 1 - self.pred
        }

    @property
    def logloss(self):
        return {
            self.t1_id: -np.log(self.pred),
            self.t2_id: -np.log(1 - self.pred)
        }

    def get_favored(self):
        if self.proba[self.t1_id] > 0.5:
            return self.t1_id
        else:
            return self.t2_id

    def get_random(self):
        if self.proba[self.t1_id] > np.random.rand():
            return self.t1_id
        else:
            return self.t2_id