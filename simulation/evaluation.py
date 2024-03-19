import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from utils import DATA_PATH


class SlotCalculator(object):
    def __init__(self, fname_slots):
        self.fname_slots = fname_slots
        self.df_slots = self._make_df_slots(self.fname_slots)
        self.set_slots = set(self.df_slots['Slot'])
        self.n_slots = self.df_slots.shape[0]
        self.dict_next_slot = self._make_dict_next_slot()
        self.dict_paths_to_victory = self._make_dict_paths_to_victory()

    @staticmethod
    def calc_round_from_slot(slot):
        if slot.startswith('R'):
            return int(slot[1])
        else:
            return 0

    @staticmethod
    def _make_df_slots(fname_slots):
        '''Reduced version of dataframe containing the slots information
        Parameters
        ----------
        fname_slots : str
            path to the file containing tournament Slots info
            Mens or Womens tournament should result in the same output here

        Returns
        -------
        df_slots : pandas DataFrame
            Slots info for NCAA tournament for Round 1 and later

        '''
        df_slots = pd.read_csv(fname_slots)

        # Only keep slots that are part of the traditional
        # tournament (no play-ins)
        df_slots = df_slots[df_slots['Slot'].str.startswith('R')]

        # except for play-ins (which we don't care about)
        # the tournament is the same structure every year
        # So, drop Season column and duplicate Slot entries
        df_slots.drop_duplicates('Slot', inplace=True)
        df_slots.drop(columns='Season', inplace=True)
        return df_slots

    def _make_dict_next_slot(self):
        '''Makes a dictionary where the value is the next Slot played by the
        team that wins the Slot specified by key.

        Returns
        -------
        next_slot : dict
        '''
        next_slot = {}
        for ir, r in self.df_slots.iterrows():
            next_slot[r['StrongSeed']] = r['Slot']
            next_slot[r['WeakSeed']] = r['Slot']
        return next_slot

    def _make_dict_paths_to_victory(self):
        '''Dictionary with paths to victory for every seed.

        Returns
        -------
        paths : dict
            Each key is a Seed in the tournament.  Each value is an ordered
            list containing the Slots that must be won by that Seed to
            win the tournament
        '''
        seeds = [f'{region}{num:02d}' for region in list('WXYZ') \
                 for num in range(1, 17)]

        paths = {}
        for s in seeds:
            slot = s
            path = []
            while slot in self.dict_next_slot.keys():
                slot = self.dict_next_slot[slot]
                path.append(slot)

            paths[s] = path
        return paths

    def calc_slot(self, seed1, seed2):
        if seed1[0:3] == seed2[0:3]:
            slot = seed1[0:3]
        else:  # check where their paths to victory intersect
            intersection = set(self.dict_paths_to_victory[seed1[0:3]]) \
                .intersection(self.dict_paths_to_victory[seed2[0:3]])
            slot = sorted(intersection)[0]
        return slot


class TournamentDataProcessor:

    def __init__(self, data_dir, gender='M'):
        self.data_dir = data_dir
        self.gender = gender
        self.df_tr = pd.read_csv(os.path.join(data_dir, f'{gender}NCAATourneyCompactResults.csv'))
        self.df_teams = pd.read_csv(os.path.join(data_dir, f'{gender}Teams.csv'))
        self.df_seeds = pd.read_csv(os.path.join(data_dir, f'{gender}NCAATourneySeeds.csv'))
        self.df_key = self._make_tourney_results_and_historical_portfolio()

    @property
    def results(self):
        return self.df_tr

    @property
    def key(self):
        return self.df_key

    def _add_missing_oregon_vcu_game_from_2021(self):
        df_mg = pd.DataFrame({'Season': [2021], 'WTeamID': [1332], 'LTeamID': [1433]})
        self.df_tr = pd.concat([self.df_tr, df_mg])
        self.df_tr.drop_duplicates(subset=['Season', 'WTeamID', 'LTeamID'], keep='first', inplace=True)
        self.df_tr.sort_values(['Season', 'DayNum'], inplace=True, ignore_index=True)

    def _add_team_name_to_tourney_results(self):
        self.df_tr = self.df_tr.merge(self.df_teams[['TeamID', 'TeamName']].rename(columns={'TeamID': 'WTeamID', 'TeamName': 'WTeamName'}), on='WTeamID', how='left')
        self.df_tr = self.df_tr.merge(self.df_teams[['TeamID', 'TeamName']].rename(columns={'TeamID': 'LTeamID', 'TeamName': 'LTeamName'}), on='LTeamID', how='left')

    def _add_seed_to_tourney_results(self):
        self.df_tr = self.df_tr.merge(self.df_seeds.rename(columns={'TeamID': 'WTeamID', 'Seed': 'WSeed'}), on=['Season', 'WTeamID'], how='left')
        self.df_tr = self.df_tr.merge(self.df_seeds.rename(columns={'TeamID': 'LTeamID', 'Seed': 'LSeed'}), on=['Season', 'LTeamID'], how='left')

    def _make_tourney_results_and_historical_portfolio(self):
        if self.gender == 'M':
            self._add_missing_oregon_vcu_game_from_2021()
        self._add_team_name_to_tourney_results()
        self._add_seed_to_tourney_results()
        self.df_tr['Tournament'] = self.gender
        slot_calculator = SlotCalculator(os.path.join(self.data_dir, 'MNCAATourneySlots.csv'))
        self.df_tr['Slot'] = [slot_calculator.calc_slot(s1, s2) for s1, s2 in zip(self.df_tr['WSeed'], self.df_tr['LSeed'])]
        self.df_tr['Round'] = [slot_calculator.calc_round_from_slot(s) for s in self.df_tr['Slot']]
        df_key = self.df_tr.loc[self.df_tr['Round'] >= 1]
        df_key = df_key[['Season', 'Tournament', 'Slot', 'WTeamName', 'LTeamName', 'WSeed']]
        df_key.rename(columns={'WSeed': 'Team'}, inplace=True)
        df_key.sort_values(['Season', 'Tournament', 'Slot'], inplace=True)
        df_key.reset_index(inplace=True, drop=True)
        df_key['Bracket'] = df_key['Season']
        df_key = df_key[['Bracket', 'Tournament', 'Slot', 'Team', 'Season', 'WTeamName', 'LTeamName']]
        df_key['Team'] = df_key['Team'].str[0:3]
        return df_key

class EvaluationProcessor:
    def __init__(self, sim_season, df_prediction=None, single=True, data_path='./data/march-machine-learning-mania-2024/'):
        m_tourney = TournamentDataProcessor(data_path, gender='M')
        w_tourney = TournamentDataProcessor(data_path, gender='W')
        self.df_combined = pd.concat([m_tourney.results, w_tourney.results])
        self.df_historical_profile = pd.concat([m_tourney.key, w_tourney.key])
        self.df_historical_profile['RowId'] = np.arange(self.df_historical_profile.shape[0])
        del m_tourney, w_tourney
        self.df_prediction = self.df_historical_profile.loc[self.df_historical_profile.Bracket == sim_season].copy() if df_prediction is None else df_prediction
        if sim_season != 2024 and single ==True:
            self.df_historical_profile = self.df_historical_profile.loc[self.df_historical_profile.Bracket == sim_season].copy()
        else:
            self.df_historical_profile = self.df_historical_profile.loc[self.df_historical_profile.Bracket <= sim_season].copy()
        self.df_evaluation = self.make_evaluation_df(self.df_prediction, self.df_historical_profile)
        self.df_seed_implied_prob = self.make_seed_implied_probability_df()

    def make_implied_probability_table(self, df_sub):
        tmp = df_sub.copy()[['Tournament', 'Slot', 'Team']].groupby(['Tournament', 'Slot']).agg('value_counts', normalize=True)
        tmp = tmp.to_frame()
        tmp.reset_index(inplace=True)
        tmp['Round'] = tmp['Slot'].str[0:2]
        tmp.drop(columns='Slot', inplace=True)
        tmp.set_index(['Tournament', 'Team', 'Round'], inplace=True)
        tmp = tmp.stack().unstack(level=2).fillna(0.0)
        tmp.reset_index(inplace=True)

        tmp.columns.name = None
        tmp.drop(columns='level_2', inplace=True)

        df_missing = []
        seeds = [f'{region}{num:02d}' for region in list('WXYZ') for num in range(1, 17)]
        for t, sdf in tmp.groupby('Tournament'):
            missing_seeds = np.setdiff1d(seeds, sdf['Team'])
            df_missing.append(pd.DataFrame({'Tournament': t, 'Team': missing_seeds}))
        df_missing = pd.concat(df_missing)
        tmp = pd.concat([tmp, df_missing])
        tmp.fillna(0.0, inplace=True)
        tmp.sort_values(['Tournament', 'Team'], inplace=True)
        tmp.reset_index(inplace=True, drop=True)
        return tmp

    def make_evaluation_df(self, pred, truth):
        proc_sub = self.make_implied_probability_table(pred)
        proc_truth = self.make_implied_probability_table(truth)
        tmp = proc_sub.merge(proc_truth, on=['Tournament', 'Team'], how='inner', suffixes=('_sub', '_truth'))
        for col in tmp.columns[tmp.columns.str.endswith('_truth')]:
            r = col.split('_')[0]
            tmp[r + '_brier'] = (tmp[r + '_sub'] - tmp[r + '_truth']) ** 2
        return tmp

    def make_seed_implied_probability_df(self):
        df_implied_prob_processed = self.make_implied_probability_table(self.df_prediction)
        df_implied_prob_processed['SeedNum'] = 'S' + df_implied_prob_processed['Team'].str[1:]
        # rename Round columns
        df_implied_prob_processed.rename(columns={r: r + 'Win' \
                                                  for r in ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']},
                                         inplace=True)
        # drop Tournament and Team columns
        df_implied_prob_processed.drop(columns=['Tournament', 'Team'], inplace=True)
        # combine SeedNum results
        return df_implied_prob_processed.groupby('SeedNum').mean()

    @property
    def submission_brier_score(self):
        brier_cols = self.df_evaluation.columns[self.df_evaluation.columns.str.endswith('_brier')]
        score = self.df_evaluation.groupby('Tournament')[brier_cols].mean().mean(axis=1).mean()
        return score

    @property
    def round_brier_score(self):
        brier_cols = self.df_evaluation.columns[self.df_evaluation.columns.str.endswith('_brier')]
        score = self.df_evaluation.groupby('Tournament')[brier_cols].mean()
        return score

    @property
    def gender_brier_score(self):
        brier_cols = self.df_evaluation.columns[self.df_evaluation.columns.str.endswith('_brier')]
        score = self.df_evaluation.groupby('Tournament')[brier_cols].mean().mean(axis=1)
        return score

    def make_evaluation_report(self):
        hp = self.df_historical_profile.drop(columns=['RowId','Bracket','Season','LTeamName']).copy()
        hp['Round'] = hp['Slot'].str[1].astype(int)

        pred = self.df_prediction.drop(columns=['RowId']).copy().rename(columns={'Team':'pred_Team'})
        a = pd.merge(pred, hp,  on=['Tournament','Slot'], how='inner')
        a = a.loc[a.Team == a.pred_Team].copy()
        round_score_mapper = {
            1: 1,
            2: 2,
            3: 4,
            4: 8,
            5: 16,
            6: 32
        }
        a['Score'] = a.Round.map(round_score_mapper)
        out = a.groupby(['Bracket','Tournament'])['Score'].sum().reset_index()
        out_r1 = a[(a.Round==1)].copy().groupby(['Bracket','Tournament'])['Score'].count().reset_index().rename(columns={'Score':'Round1GamesCorrect'})
        out_r5 = a[(a.Round==5)].copy().groupby(['Bracket','Tournament'])['Score'].count().reset_index().rename(columns={'Score':'Final4Correct'})
        out_r6 = a[(a.Round==6)].copy().groupby(['Bracket','Tournament'])['Score'].count().reset_index().rename(columns={'Score':'ChampCorrect'})
        out = pd.merge(out, out_r1,on=['Bracket','Tournament'], how='left')
        out = pd.merge(out, out_r5,on=['Bracket','Tournament'], how='left')
        out = pd.merge(out, out_r6,on=['Bracket','Tournament'], how='left')
        out = out.fillna(0)
        out.Round1GamesCorrect = out.Round1GamesCorrect.astype(int)
        out.Final4Correct = out.Final4Correct.astype(int)
        out.ChampCorrect = out.ChampCorrect.astype(int)

        #### Questions
        evaluation_report = {}
        ## - What was our submission brier score?
        evaluation_report['SubmissionBrierScore'] = self.submission_brier_score
        ## - How many times did we get the champ right?
        evaluation_report['ChampCorrectCount'] = out.groupby(['Tournament'])['ChampCorrect'].sum().to_dict()
        ## - How many final 4 teams did we get right on average?
        evaluation_report['Final4CorrectAvg'] = out.groupby(['Tournament'])['Final4Correct'].mean().to_dict()
        ## - How many round 1 teams did we get right on average?
        evaluation_report['Round1CorrectAvg'] = out.groupby(['Tournament'])['Round1GamesCorrect'].mean().to_dict()
        ## - How many round 1s did we get perfect?
        evaluation_report['PerfectRound1Count'] = out[(out.Round1GamesCorrect==32)].groupby(['Tournament'])['Round1GamesCorrect'].count().to_dict()
        ## - What was our average score ?
        evaluation_report['BracketScoreAvg'] = out.groupby(['Tournament'])['Score'].mean().to_dict()
        ## - What was our max score ?
        evaluation_report['BracketScoreMax'] = out.groupby(['Tournament'])['Score'].max().to_dict()
        ## - What was our min score ?
        evaluation_report['BracketScoreMin'] = out.groupby(['Tournament'])['Score'].min().to_dict()
        ## - How many perfect brackets did we generate?
        evaluation_report['PerfectBracketCount'] = out[(out.Score==192)].groupby(['Tournament'])['Score'].count().to_dict()
        ## - What was our gender brier score split?
        evaluation_report['GenderBrierScore'] = self.gender_brier_score.to_dict()
        ## - What was our round brier score split?
        evaluation_report['RoundBrierScore'] = self.round_brier_score.to_dict()
        return evaluation_report


def make_64_seeds(sim_season, gender):
    tourney = TournamentDataProcessor(DATA_PATH, gender=gender)
    seeds_df = tourney.results.copy()
    seeds_df = seeds_df.loc[seeds_df.Season == sim_season].copy()
    wseeds_df = seeds_df.loc[seeds_df.Round == 1][['Season','WTeamID','WSeed']]
    lseeds_df = seeds_df.loc[seeds_df.Round == 1][['Season','LTeamID','LSeed']]
    wseeds_df.columns = ['Season','TeamID','Seed']
    lseeds_df.columns = ['Season','TeamID','Seed']
    seeds_df = pd.concat([wseeds_df, lseeds_df])
    seeds_df.Seed = seeds_df.Seed.str[0:3]
    return seeds_df
