import pandas as pd

from simulation.prediction import Prediction
from simulation.utils import get_alt_game_id


class Submission:
    '''
    Submission is a container for the Prediction class and with
    a method that helps to retrieve predictions based on
    the game_id.
    '''

    def __init__(self, sub_df, data):
        sub_df[['Season', 'Team1ID', 'Team2ID']] = \
            sub_df['ID'].str.split('_', expand=True)
        sub_df[['Season', 'Team1ID', 'Team2ID']] = \
            sub_df[['Season', 'Team1ID', 'Team2ID']].astype(int)
        sub_df['Round'] = \
            sub_df.apply(
                lambda row:
                data.get_round(row['Season'],
                               row['Team1ID'],
                               row['Team2ID']),
                axis=1
            )

        self.seasons = sub_df['Season'].unique().tolist()
        self._df = sub_df.copy()
        self.t_dict = data.t_dict
        self.t_dict_rev = data.t_dict_rev

        def prediction_init(row):
            s_dict = data.seedyear_dict[row['Season']]
            pred = Prediction(row['Season'],
                              row['Team1ID'],
                              row['Team2ID'],
                              row['Pred'],
                              self.t_dict,
                              s_dict)
            return pred

        self._df['PredData'] = self._df.apply(prediction_init, axis=1)

    def get_pred(self, game_id=None):
        '''
        Retrieve prediction using game_id or leave blank to get
        all predictions in a list
        '''
        if game_id is None:
            return self.predictions

        alt_game_id = get_alt_game_id(game_id)
        sub_ids = self.predictions.apply(lambda x: x.game_id)

        idx = ((sub_ids == game_id) |
               (sub_ids == alt_game_id))

        if idx.sum() == 0:
            raise ValueError('Game not found!')

        pred = self.predictions.loc[idx].squeeze()
        return pred

    def get_pred_by_teams(self,
                          season=2021,
                          t1_id=None,
                          t2_id=None,
                          t1_name=None,
                          t2_name=None,):
        ids = False
        if t1_id is not None and t2_id is not None:
            ids = True
        elif t1_name is not None and t2_name is not None:
            if ids:
                raise ValueError(
                    'provide only names or ids of team'
                    )
            t1_id = self.t_dict_rev.get(t1_name)
            t2_id = self.t_dict_rev.get(t2_name)
        else:
            raise ValueError(
                'Please provide a name or ID for both team 1 and 2'
            )
        game_id = f'{season}_{t1_id}_{t2_id}'
        pred = self.lookup_df.loc[game_id, 'PredData']
        return pred

    @property
    def predictions(self):
        return self.df['PredData']

    @property
    def df(self):
        df = self._df.copy()
        df.set_index('ID', inplace=True)
        col_order = ['Season', 'Round', 'Team1ID',
                     'Team2ID', 'Pred', 'PredData']

        return df[col_order]

    @property
    def lookup_df(self):
        df = self.df.copy()
        df_swap = df.copy()
        df_swap.index = df_swap['PredData'].map(
            lambda x: x.alt_game_id
            )
        df_swap.index.name = 'ID'
        df_swap[['Team1ID', 'Team2ID']] = \
            df[['Team2ID', 'Team1ID']].values
        df_swap['Pred'] = 1 - df_swap['Pred']

        return pd.concat([df, df_swap])