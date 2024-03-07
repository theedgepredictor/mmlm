import graphviz
import numpy as np
import pandas as pd

from simulation.game import Game
from simulation.team import Team
from simulation.utils import ROUND_NAMES


class Tournament:

    def __init__(self, data, submission, season):
        '''
        Tournament class will hold all game classes and functions
        to handle games. Needs NCAA data and submission as input
        along with the season that will be modeled.
        '''

        # Add metadata to be called by class
        self.submission = submission  # submission class to get preds
        self.mw = data.mw
        self.season = season  # season year
        self.current_r = 0  # initiate at round 0 (play-in)
        self.results = {}  # results stored as slot: TeamID

        # Create seed: teamID dictionary
        self.t_dict = data.t_dict
        self.s_dict = data.seedyear_dict[self.season]
        self.s_dict_rev = data.seedyear_dict_rev[self.season]
        self._summary = {}

        # Only men's file has differing slots by year - select the year
        #       we need and remove season column
        slots = data.slots
        if self.mw == 'M':
            slots = slots[data.slots['Season'] == season].copy()
            slots.drop(columns='Season', inplace=True)
        else:
            if season < 2022:
                slots = slots[0]
            else:
                slots = slots[1]

        # Initiate game classes and save as Tournament attribute
        def game_init(row):
            game = Game(row, data.t_dict,
                        self.s_dict, self.season)
            return game

        if (len(self.s_dict) == 0) or (len(slots) == 0):
            raise RuntimeError('''
                    Please check to see that your submission file and
                    tournament data class has both have the appropriate season.
                    ''')
        self.games = slots.apply(game_init, axis=1)
        self.games.index = slots['Slot']

    @property
    def n_teams(self):
        return len(self.s_dict)

    @property
    def summary(self):
        if len(self._summary) == 0:
            self._summary = self.summarize_results()
        return self._summary

    def summarize_results(self, previous_summary=None):
        if previous_summary is not None:
            self._summary = previous_summary
        for slot, team in self.results.items():
            team = team.id
            r = slot[:2]
            if 'R' not in r:
                r = 'R0'
            if self._summary.get(r) is None:
                self._summary.update({r: {team: 1}})
            elif self._summary[r].get(team) is None:
                self._summary[r].update({team: 1})
            else:
                self._summary[r][team] += 1
        return self._summary

    def summary_to_df(self, summary=None, n_sim=1):

        if summary is None:
            summary = self.summary

        columns = [ROUND_NAMES.get(k) for k in range(7)]
        if self.mw == 'W' and self.season < 2022:
            columns = columns[1:]
        summary_df = pd.DataFrame(summary)
        summary_df.columns = columns
        summary_df.index.name = 'TeamID'
        all_teams = list(self.s_dict.values())
        missing_teams = list(set(all_teams) - set(summary_df.index))
        if len(missing_teams) > 0:
            missing_teams_df = pd.DataFrame(np.nan,
                                            index=missing_teams,
                                            columns=summary_df.columns)
            summary_df = pd.concat([summary_df, missing_teams_df])
        summary_df['Team'] = [f'{self.s_dict_rev[t]} - '
                              f'{self.submission.t_dict[t]}'
                              for t in summary_df.index]
        columns.insert(0, 'Team')
        summary_df = summary_df[columns]

        summary_df['First Round'].fillna(n_sim, inplace=True)
        summary_df.fillna(0, inplace=True)
        summary_df.sort_values(by=columns[::-1], ascending=False, inplace=True)
        return summary_df

    def simulate_games(self, style):
        '''
        This function uses each game class from a specific round
        to predict the winner (set to either the favorite or
        a random sample based on the odds). The winner is added to
        the results file under the appropriate tournament slot.
        To advance teams to the next round us advance_teams()
        afer this or run simulate_round(), which runs both.
        '''

        # function pull predicted result from game if in same round
        def find_winner(x):
            if x.r == self.current_r:
                win_id = x.get_winner(self.submission, style)
                if x.strong_team.id == win_id:
                    self.results.update({x.slot: x.strong_team})
                elif x.weak_team.id == win_id:
                    self.results.update({x.slot: x.weak_team})
                else:
                    raise ValueError('Couldn\'t find winner')

            else:
                pass

        self.games.apply(find_winner)  # apply function

    def advance_teams(self):
        '''
        calls on all tournament games to update their slots
        based on results dictionary
        '''

        self.games.apply(lambda x: x.add_teams(self.results))

    def simulate_round(self, style):
        '''
        runs the appropriate functions to both simulate the
        current round and advances teams. Note that round needs
        to be manually incremented and can be found in
        Tournament.current_r. If you want to sim the whole tourney
        and you don't need to alter data in between rounds just
        use simulate_tournament()
        '''

        self.simulate_games(style)
        self.advance_teams()

    def simulate_tournament(self, style, seed=None):
        '''
        Runs single round simulation until all are complete.
        '''
        if seed is not None:
            np.random.seed(seed)  # seed np at tournament level

        # Run simulations for round 0->6
        while self.current_r < 7:
            self.simulate_round(style)
            self.current_r += 1  # increments round by 1

    def simulate_tournaments(self, n_sim=500):
        '''
        Puts the tournament results in a summary format keyed
        by team and round that can be aggregated over multiple
        simulations. This results dict can be made into a pandas
        dataframe by simply calling pd.DataFrame(results) on
        a results dictionary that holds simulated outputs.
        args:
            summary_dict: (dict) if running multiple simultaions
                put the previous summary result as an argument
                to iteratively
        '''

        summary = {}
        expected_losses = []

        for i in range(int(n_sim)):
            self.reset_tournament()
            self.simulate_tournament('random', seed=i)
            summary = self.summarize_results(previous_summary=summary)
            losses = self.get_losses(kaggle=True)
            loss = losses.mean()
            expected_losses.append(loss)

        self._summary = summary
        self.expected_losses = np.array(expected_losses)
        return self.summary_to_df(self._summary, n_sim=n_sim), \
            self.expected_losses

    def get_losses(self, kaggle=True):
        '''
        gets losses for all predictions based on the results
        dictionary

        Kaggle=True to exlude play-ins
        '''

        def logloss(x):
            w_id = self.results.get(x.slot).id
            if w_id is None:
                return np.nan()
            game_id = x.game_id
            pred = self.submission.get_pred(game_id)
            logloss = pred.logloss[w_id]
            return logloss

        losses = self.games.apply(lambda x: logloss(x))
        if kaggle:
            losses = losses.loc[
                    losses.index.str.startswith('R')
                    ]

        return losses

    def get_odds(self, kaggle=True):
        '''
        gets odds for all predictions based on the results
        dictionary
        '''

        def calc_odds(x):
            w_id = self.results.get(x.slot).id
            if w_id is None:
                return np.nan()
            game_id = x.game_id
            pred = self.submission.get_pred(game_id)
            proba = pred.proba[w_id]
            return proba

        odds = self.games.apply(lambda x: calc_odds(x))
        if kaggle:
            odds = odds.loc[
                    odds.index.str.startswith('R')
                    ]
        return odds

    def graph_games(self, rounds=list(range(7))):
        games = [g for g in self.games if g.r in rounds]

        graph = graphviz.Digraph(node_attr={
            'shape': 'rounded',
            'color': 'lightblue2'
        })
        for g in games:

            T1 = 'R' + f'{g.r} {g.strong_team.seed}-{g.strong_team.name}'
            T2 = 'R' + f'{g.r} {g.weak_team.seed}-{g.weak_team.name}'
            W = 'R' + f'{g.r+1} {self.results[g.slot].seed}' \
                f'-{self.results[g.slot].name}'

            pred = self.submission.get_pred(f'{self.season}_'
                                            f'{g.strong_team.id}_'
                                            f'{g.weak_team.id}')
            if g.strong_team.name == self.results[g.slot].name:
                odds = pred.proba[g.strong_team.id]
                T1_params = {'color': 'green', 'label': f'{odds:.0%}'}
                T2_params = {'color': 'red'}

            else:
                odds = pred.proba[g.weak_team.id]
                T2_params = {'color': 'green', 'label': f'{odds:.0%}'}
                T1_params = {'color': 'red'}

            graph.edge(T1, W, **T1_params)
            graph.edge(T2, W, **T2_params)

        graph.graph_attr['rankdir'] = 'LR'
        graph.graph_attr['size'] = '30'

        graph.node_attr.update(style='rounded')

        return graph

    def update_results(self, new_results):
        '''
        method to update the results dict with a generic
        slots: team_id dict
        '''
        self.reset_tournament()
        new_results_team = \
            {slot: Team(t_id=tid,
                        name=self.t_dict.get(tid),
                        seed=self.s_dict_rev.get(tid)
                        )
             for slot, tid in new_results.items()}
        self.results.update(new_results_team)

        self.advance_teams()
        self._summary = {}

    def reset_tournament(self):
        self.current_r = 0  # initiate at round 0 (play-in)
        self.results = {}  # results stored as slot: TeamID
        self._summary = {}