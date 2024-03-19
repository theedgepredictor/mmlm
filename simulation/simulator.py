import argparse
import json
import os

import numpy as np
import pandas as pd

from experiments.elo import EloGame
from utils import DATA_PATH
from tqdm import tqdm

from simulation.evaluation import EvaluationProcessor, make_64_seeds


class QuickTourneySimulator:
    def __init__(self, seeds, preds, round_slots, ratings=None):
        self.seeds = seeds
        self.preds = preds
        self.round_slots = round_slots
        if ratings is not None:
            self.ratings = ratings
        else:
            self.ratings = None
        self.seed_dict = self.seeds.set_index('Seed')['TeamID'].to_dict()
        self.inverted_seed_dict = {value: key for key, value in self.seed_dict.items()}
        self.probas_dict, self.rating_dict = self._prepare_data()
        self.base_ratings_dict = self.rating_dict

    def _prepare_data(self):
        if self.ratings is not None:
            rating_dict = {int(key): val for key, val in self.ratings.items()}
        else:
            rating_dict = None
        probas_dict = {}
        if self.preds is not None:
            for teams, proba in zip(self.preds['ID'], self.preds['Pred']):
                team1, team2 = teams[1], teams[2]
                probas_dict.setdefault(team1, {})[team2] = proba
                probas_dict.setdefault(team2, {})[team1] = 1 - proba
        return probas_dict, rating_dict

    def _simulate_model_prob_game(self, slot, strong, weak, random_val, sim=True):
        team_1, team_2 = self.seed_dict[strong], self.seed_dict[weak]

        proba = self.probas_dict[str(team_1)][str(team_2)]
        meta_row = {
            'str_event_id': slot,
            'season': 1,
            'date': pd.Timestamp('1900-01-01'),
            'neutral_site': 1,
            'home_team_name': team_1,
            'away_team_name': team_2,
            'proba': proba
        }

        if sim:
            # Randomly determine the winner based on the probability
            winner = team_1 if random_val < proba else team_2
        else:
            # Determine the winner based on the higher probability
            winner = [team_1, team_2][np.argmax([proba, 1 - proba])]

        meta_row['winning_seed'] = winner
        meta_row['winning_team_id'] = winner
        meta_row['home_seed'] = strong
        meta_row['away_seed'] = weak
        meta_row['proba'] = proba
        return meta_row

    def _simulate_team_point_estimate(self, slot, strong, weak, pred_dict, random_val):
        team_1, team_2 = self.seed_dict[strong], self.seed_dict[weak]

        pred_row = pred_dict[f"{team_1}_{team_2}"]
        team1_pred_score = pred_row[f"{team_1}"]
        team2_pred_score = pred_row[f"{team_2}"]

        ### Add model error rate as randomizer bounds
        mae = 7  # 13.68 / 2
        t1_score_options = list(range(int(team1_pred_score - mae), int(team1_pred_score + mae)))
        t2_score_options = list(range(int(team2_pred_score - mae), int(team2_pred_score + mae)))
        team1_adj_pred_score = t1_score_options[random_val]
        team2_adj_pred_score = t2_score_options[np.random.randint(0,13)]


        meta_row = {
            'str_event_id': slot,
            'season': 1,
            'date': pd.Timestamp('1900-01-01'),
            'neutral_site': 1,
            'home_team_name': team_1,
            'home_team_score': team1_adj_pred_score,
            'away_team_name': team_2,
            'away_team_score': team2_adj_pred_score,
        }

        winner = team_1 if team1_adj_pred_score >= team2_adj_pred_score else team_2

        meta_row['winning_seed'] = winner
        meta_row['winning_team_id'] = winner
        meta_row['home_seed'] = strong
        meta_row['away_seed'] = weak
        return meta_row

    def _simulate_iterative_elo_game(self, slot, strong, weak, random_val, sim=True):
        team_1, team_2 = self.seed_dict[strong], self.seed_dict[weak]
        if self.rating_dict is not None:
            team1_rating = self.rating_dict[team_1]
            team2_rating = self.rating_dict[team_2]

            ### Add model prediction calculation here from current elos

            ### For sample lets use last years model prior to spline
            ### where the higher pred score will be score_diff + season_avg_score (like 65 ish)
            ### the lower pred score will be the season_avg_score
            ### Ex: Score Dif: -8 team1: 65 team2: 65+8=73
            ### But this will allow us to start looking at team1 vs team2 pred scores instead of just
            ### an overall score diff or just a probability
            avg_points_per_game_all_time = 65  # 65.28
            spread = -(team1_rating - team2_rating) / 30

            team1_pred_score = avg_points_per_game_all_time
            team2_pred_score = avg_points_per_game_all_time + spread

            ### Add model error rate as randomizer bounds
            mae = 7  # 13.68 / 2
            if team1_pred_score is not None and team2_pred_score is not None:

                t1_score_options = list(range(int(team1_pred_score - mae), int(team1_pred_score + mae)))
                t2_score_options = list(range(int(team2_pred_score - mae), int(team2_pred_score + mae)))
                team1_adj_pred_score = np.random.choice(t1_score_options)
                team2_adj_pred_score = np.random.choice(t2_score_options)
            else:
                team1_adj_pred_score = None
                team2_adj_pred_score = None

            meta_row = {
                'str_event_id': slot,
                'season': 1,
                'date': pd.Timestamp('1900-01-01'),
                'neutral_site': 1,
                'home_team_name': team_1,
                'home_team_score': team1_adj_pred_score,
                'away_team_name': team_2,
                'away_team_score': team2_adj_pred_score,
                'home_elo_pre': team1_rating,
                'home_elo_prob': None,
                'home_elo_post': None,
                'away_elo_pre': team2_rating,
                'away_elo_prob': None,
                'away_elo_post': None,
            }
            elo_game = EloGame(**meta_row)
            res = elo_game.sim(
                k=30,
                hfa=100,
                width=800,
                allow_future=True
            )
            # Update new pre ratings
            if res['home_elo_post'] is not None and res['away_elo_post'] is not None:
                self.rating_dict[team_1] = res['home_elo_post']
                self.rating_dict[team_2] = res['away_elo_post']

            # update metadata record
            meta_row['home_elo_prob'] = res['home_elo_prob']
            meta_row['home_elo_post'] = res['home_elo_post']
            meta_row['away_elo_prob'] = res['away_elo_prob']
            meta_row['away_elo_post'] = res['away_elo_post']

            meta_row['home_pre_adj_score'] = team1_pred_score
            meta_row['away_pre_adj_score'] = team2_pred_score

            # Get the probability of team_1 winning
            proba = res['home_elo_prob']
            meta_row['proba'] = proba
        else:
            proba = self.probas_dict[str(team_1)][str(team_2)]
            meta_row = {
                'str_event_id': slot,
                'season': 1,
                'date': pd.Timestamp('1900-01-01'),
                'neutral_site': 1,
                'home_team_name': team_1,
                'home_team_score': None,
                'away_team_name': team_2,
                'away_team_score': None,
                'home_elo_pre': None,
                'home_elo_prob': None,
                'home_elo_post': None,
                'away_elo_pre': None,
                'away_elo_prob': None,
                'away_elo_post': None,
                'home_pre_adj_score': None,
                'away_pre_adj_score': None,
                'proba': proba
            }

        if sim and (meta_row['home_pre_adj_score'] is not None and meta_row['away_pre_adj_score'] is not None):
            # Randomly determine the winner based on the probability
            winner = team_1 if random_val < proba else team_2
        else:
            # Determine the winner based on the higher probability
            winner = [team_1, team_2][np.argmax([proba, 1 - proba])]

        meta_row['winning_seed'] = winner
        meta_row['winning_team_id'] = winner
        meta_row['home_seed'] = strong
        meta_row['away_seed'] = weak
        meta_row['proba'] = proba
        return meta_row

    def _simulate_benchmark_game(self, slot, seed1, seed2, random_val, sim=True):
        '''Simple traditional linear model for the probability that treats seeds like integers
        Not a great model, but it's ok and about as easy as it gets
        '''
        # prob that seed1 wins
        team_1, team_2 = self.seed_dict[seed1], self.seed_dict[seed2]

        int_seed1 = int(self.inverted_seed_dict[team_1][1:3])

        int_seed2 = int(self.inverted_seed_dict[team_2][1:3])


        proba = 0.5 + 0.03 * (int_seed2 - int_seed1)
        if sim:
            winner = team_1 if random_val < proba else team_2
        else:
            winner = [team_1, team_2][np.argmax([proba, 1 - proba])]
        meta_row = {
            'str_event_id': slot,
            'season': 1,
            'date': pd.Timestamp('1900-01-01'),
            'neutral_site': 1,
            'home_team_name': team_1,
            'home_team_score': None,
            'away_team_name': team_2,
            'away_team_score': None,
            'winning_seed': winner,
            'winning_team_id': winner,
            'home_seed': seed1,
            'away_seed': seed2,
            'proba': proba
        }
        return meta_row

    def simulate(self, random_values, experiment, pred_dict = None):
        winners = []
        slots = []
        metadata = []
        if experiment == 'elo':
            # Reset elo ratings when a new bracket gets simulated
            self.rating_dict = self.base_ratings_dict
        for slot, strong, weak, random_val in zip(self.round_slots.Slot, self.round_slots.StrongSeed, self.round_slots.WeakSeed, random_values):
            if experiment == 'elo':
                meta_row = self._simulate_iterative_elo_game(slot, strong, weak, random_val=random_val, sim=True)
            elif experiment in ['madness-2023-1st', 'madness-2024-copy','madness-2024-sub-proba']:
                meta_row = self._simulate_model_prob_game(slot, strong, weak, random_val=random_val, sim=True)
            elif experiment in ['madness-2024-sub']:
                meta_row = self._simulate_team_point_estimate(slot, strong, weak, pred_dict, random_val=random_val)
            elif experiment == 'benchmark':
                meta_row = self._simulate_benchmark_game(slot, strong, weak, random_val=random_val)
            else:
                raise Exception('Invalid Experiment')

            winners.append(meta_row['winning_team_id'])
            slots.append(slot)

            self.seed_dict[slot] = meta_row['winning_team_id']
            metadata.append(meta_row)

        return [self.inverted_seed_dict[w] for w in winners], slots, metadata

    def run_simulation(self, brackets=1, experiment='benchmark'):
        results = []
        bracket = []
        slots = []
        metadata = []

        if experiment == 'madness-2024-sub':
            random_values = np.random.randint(0,14, size=(brackets, len(self.round_slots)))
            pred_dict = {}
            for teams, pred1, pred2 in zip(self.preds['ID'], self.preds['T1Pred'], self.preds['T2Pred']):
                team1, team2 = teams[1], teams[2]
                pred_dict[f"{team1}_{team2}"] = {team1: pred1, team2: pred2}
                pred_dict[f"{team2}_{team1}"] = {team2: pred1, team1: pred2}
        else:
            random_values = np.random.random(size=(brackets, len(self.round_slots)))
            pred_dict = None

        for b in tqdm(range(1, brackets + 1)):
            r, s, m = self.simulate(random_values[b - 1], experiment, pred_dict)
            results.extend(r)
            bracket.extend([b] * len(r))
            slots.extend(s)
            metadata.extend(m)

        result_df = pd.DataFrame({'Bracket': bracket, 'Slot': slots, 'Team': results})
        return result_df, pd.DataFrame(metadata)


def runner(sim_season, experiment, n_brackets=5000, predictions_prefix='./output',data_path='./data/march-machine-learning-mania-2024/'):
    prediction_path = f'{predictions_prefix}/{experiment}/{sim_season}/'

    if not experiment == 'benchmark':
        try:
            preds = pd.read_csv(f'{prediction_path}predictions.csv')
        except Exception as e:
            print(f'Need to run experiment: {experiment} for {sim_season}')
            if experiment == 'elo':
                from experiments.elo import runner as elo_runner
                elo_runner(sim_season)
            elif experiment == 'madness-2023-1st':
                from experiments.madness_2023_1st import runner as madness_2023_1st_runner
                madness_2023_1st_runner(sim_season, repeat_cv=10, upset_overrides=True)
            elif experiment in ['madness-2024-sub','madness-2024-sub-proba']:
                from experiments.madness_2024_sub import runner as madness_2024_sub_runner
                madness_2024_sub_runner(sim_season, repeat_cv=10, upset_overrides=True)
            else:
                raise Exception("Invalid Experiment")
            preds = pd.read_csv(f'{prediction_path}predictions.csv')
        preds['ID'] = preds['ID'].str.split('_')
    else:
        preds = None
    if sim_season != 2024:
        seeds_m = make_64_seeds(sim_season,'M')
        seeds_w = make_64_seeds(sim_season,'W')
    else:
        new_seeds = pd.read_csv(data_path + "2024_tourney_seeds.csv")
        new_seeds['Season'] = 2024
        seeds_m = new_seeds.loc[new_seeds.Tournament=='M'].drop(columns='Tournament').copy()
        seeds_w = new_seeds.loc[new_seeds.Tournament=='W'].drop(columns='Tournament').copy()

    round_slots = pd.read_csv(data_path + 'MNCAATourneySlots.csv')
    round_slots = round_slots[round_slots['Season'] == sim_season]
    round_slots = round_slots[round_slots['Slot'].str.contains('R')] # Filter out First Four

    has_elo = experiment in ['elo']

    if has_elo:
        with open(f"{prediction_path}latest_elo.json", 'r') as file:
            ratings = json.load(file)
    else:
        ratings = None


    m_qts = QuickTourneySimulator(
        seeds=seeds_m,
        preds=preds,
        round_slots=round_slots,
        ratings=ratings
    )
    result_m, meta_m = m_qts.run_simulation(
        brackets=n_brackets,
        experiment=experiment
    )
    result_m['Tournament'] = 'M'
    w_qts = QuickTourneySimulator(
        seeds=seeds_w,
        preds=preds,
        round_slots=round_slots,
        ratings=ratings
    )
    result_w, meta_w = w_qts.run_simulation(
        brackets=n_brackets,
        experiment=experiment
    )
    result_w['Tournament'] = 'W'
    submission = pd.concat([result_m, result_w])
    submission['RowId'] = np.arange(submission.shape[0])
    submission = submission.reset_index()
    if 'level_0' in submission.columns:
        submission = submission.drop(columns='level_0')
    if 'index' in submission.columns:
        submission = submission.drop(columns='index')
    meta = pd.concat([meta_m, meta_w])
    eval = EvaluationProcessor(sim_season, submission, data_path=data_path)

    os.makedirs(prediction_path, exist_ok=True)
    if sim_season != 2024:
        print('EVALUATION REPORT')
        print()
        evaluation_report = eval.make_evaluation_report()
        print(evaluation_report)
        with open(prediction_path+'evaluation_report.json', 'w') as json_file:
            json.dump(evaluation_report, json_file, indent=2)
    submission_cols = ['RowId','Tournament','Bracket','Slot','Team']
    submission[submission_cols].to_csv(f"{prediction_path}submission.csv", index=False)
    meta.to_csv(f"{prediction_path}submission_metadata.csv", index=False)
    return submission[submission_cols]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMLM Simulator')
    parser.add_argument('sim_season', type=int, help='The season to generate predictions for')
    parser.add_argument('--experiment', type=str, default='madness-2023-1st', help='The experiment to run')
    parser.add_argument('--n_brackets', type=int, default=10, help='Number of brackets to sim')
    args = parser.parse_args()
    runner(args.sim_season, args.experiment, args.n_brackets)
