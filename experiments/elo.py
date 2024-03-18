import argparse
import json
import math
import os

import numpy as np
import pandas as pd

from utils import df_rename_fold, is_pandas_none, DATA_PATH

initial_load_columns = ['str_event_id', 'season', 'date', 'neutral_site', 'home_team_name', 'home_team_score', 'away_team_name', 'away_team_score']
upsert_load_columns = initial_load_columns + ['home_elo_pre', 'away_elo_pre', 'home_elo_prob', 'away_elo_prob', 'home_elo_post', 'away_elo_post']

ELO_SCHEMA = {
    'id': np.int64,
    'season': np.int32,
    'is_postseason': np.int8,
    'tournament_id': 'Int32',
    'is_finished': np.int8,
    'neutral_site': np.int8,
    'home_team_id': np.int32,
    'home_team_score': 'Int32',
    'away_team_id': np.int32,
    'away_team_score': 'Int32',
    'home_elo_pre': np.float64,
    'away_elo_pre': np.float64,
    'home_elo_prob': np.float64,
    'away_elo_prob': np.float64,
    'home_elo_post': np.float64,
    'away_elo_post': np.float64
}


class EloRunner:
    """
    Base Elo Runner for any 1v1 event

    Attributes:
        runner_df (pd.DataFrame): DataFrame for the EloRunner.
        current_elos (dict): Dictionary containing current Elo ratings for teams.
        games (list): List to store EloGame simulation results.
        mode (str): Mode of the EloRunner ('refresh' or 'upsert').
        allow_future (bool): Flag to include future events in the simulation.
        _k (int): K Factor. Higher K = higher rating change.
        _mean_elo (int): Average rating score of the system.
        _hfa (int): Home team is awarded this many points to their base rating.
        _width (int): Lower and upper bounds of Elo ratings (mean_elo - width, mean_elo + width).
        _revert_percentage (float): Percentage of regression towards the mean. (common is 1/3 revert back to mean)
        preloaded_elos (dict): Dictionary of preloaded Elo ratings.

    Methods:
        _load_state(df, preloaded_elos=None): Load initial or upsert state and preloaded Elo ratings.
        run_to_date(): Run Elo simulations for each event up to the current date.
        rating_reset(): Regression towards the mean for team ratings.
    """

    def __init__(
            self, df: pd.DataFrame(),
            mode: str = 'refresh',
            allow_future: bool = False,
            k: int = 20,
            mean_elo: int = 1500,
            home_field_advantage: int = 100,
            width: int = 400,
            revert_percentage: float = 1.0 / 3,
            preloaded_elos=None
    ):
        """
        Initialize EloRunner.

        Args:
            df (pd.DataFrame): DataFrame for EloRunner.
            mode (str): Mode of EloRunner ('refresh' or 'upsert').
            allow_future (bool): Flag to include future events in simulation.
            k (int): K Factor. Higher K = higher rating change.
            mean_elo (int): Average rating score.
            home_field_advantage (int): Home field advantage in Elo ratings.
            width (int): Lower and upper bounds of Elo ratings (mean_elo - width, mean_elo + width).
            revert_percentage (float): Percentage of regression towards the mean. (common is 1/3 revert back to mean)
            preloaded_elos (dict): Dictionary of preloaded Elo ratings.
        """
        self.runner_df = pd.DataFrame()
        self.current_elos = {}
        self.games = []
        self.mode = mode
        self.allow_future = allow_future
        self._k = k
        self._mean_elo = mean_elo
        self._hfa = home_field_advantage
        self._width = width
        if revert_percentage > 1 or revert_percentage < 0:
            raise Exception('Invalid revert percentage')
        self._revert_percentage = revert_percentage

        self._load_state(df.copy(), preloaded_elos=preloaded_elos)

    def _load_state(self, df, preloaded_elos=None):
        """
        Load initial or upsert state and preloaded Elo ratings.

        Args:
            df (pd.DataFrame): DataFrame for EloRunner.
            preloaded_elos (dict): Dictionary of preloaded Elo ratings.
        """
        if len(df.columns) == len(initial_load_columns):
            df = df[initial_load_columns].copy()
        elif len(df.columns) == len(upsert_load_columns):
            df = df[upsert_load_columns].copy()
        else:
            raise Exception('Invalid DataFrame Dimensions')

        if 'home_elo_pre' in df.columns and 'away_elo_pre' in df.columns:
            if len(df.loc[~df.home_elo_pre.isnull()].home_elo_pre.values) > 0 and len(df.loc[~df.away_elo_pre.isnull()].away_elo_pre.values) > 0:
                self.mode = 'upsert'
        else:
            df['home_elo_pre'] = None
            df['away_elo_pre'] = None
            df['home_elo_prob'] = None
            df['away_elo_prob'] = None
            df['home_elo_post'] = None
            df['away_elo_post'] = None

        df = df[upsert_load_columns]
        df['date'] = pd.to_datetime(df['date'])
        df['neutral_site'] = df['neutral_site'].astype(int)

        unique_teams = list(set(list(df.home_team_name.values) + list(df.away_team_name.values)))
        self.current_elos = dict(zip(unique_teams, [self._mean_elo for _ in unique_teams]))
        df = df.sort_values(['season', 'date'])

        if preloaded_elos is not None:
            self.current_elos = {**self.current_elos, **preloaded_elos}
            self.runner_df = df
        elif self.mode == 'upsert':
            # Save default elos in case there are teams that do not have a previous elo rating (new team during update)
            default_elos = self.current_elos
            # Determine games we need to run and save that subset as the runner_df
            latest_df = df.loc[(
                    (df.home_team_score.notnull()) &
                    (df.away_team_score.notnull()) &
                    (df.away_elo_pre.notnull()) &
                    (df.home_elo_pre.notnull())
            )]
            # Get latest elo for each team
            latest_df = df_rename_fold(latest_df, 'away_', 'home_')
            team_latest_elos = latest_df.sort_values('date').groupby('team_name')['elo_post'].last().reset_index()
            latest_elos = dict(zip(list(team_latest_elos.team_name.values), list(team_latest_elos.elo_post.values)))
            self.current_elos = {**default_elos, **latest_elos}
            df = df.sort_values(['season', 'date'])
            self.runner_df = df.loc[~(
                    (df.home_team_score.notnull()) &
                    (df.away_team_score.notnull()) &
                    (df.away_elo_pre.notnull()) &
                    (df.home_elo_pre.notnull())
            )]
            if self.runner_df.season.min() != latest_df.season.min():
                self.rating_reset()
        else:
            self.runner_df = df

    def run_to_date(self):
        """
        Run Elo simulations for each event up to the current date.

        Returns:
            pd.DataFrame: DataFrame containing Elo simulation results.
        """
        current_season = self.runner_df.season.min()
        for row in self.runner_df.itertuples(index=False):
            if row.season != current_season:
                self.rating_reset()
                current_season = row.season
            dict_row = {
                'str_event_id': row.str_event_id,
                'season': row.season,
                'date': row.date,
                'neutral_site': row.neutral_site,
                'home_team_name': row.home_team_name,
                'home_team_score': row.home_team_score,
                'away_team_name': row.away_team_name,
                'away_team_score': row.away_team_score,
                'home_elo_pre': self.current_elos[row.home_team_name],
                'home_elo_prob': row.home_elo_prob,
                'home_elo_post': row.home_elo_post,
                'away_elo_pre': self.current_elos[row.away_team_name],
                'away_elo_prob': row.away_elo_prob,
                'away_elo_post': row.away_elo_post,
            }
            elo_game = EloGame(**dict_row)
            res = elo_game.sim(
                k=self._k,
                hfa=self._hfa,
                width=self._width,
                allow_future=self.allow_future
            )
            if res['home_elo_post'] is not None and res['away_elo_post'] is not None:
                self.current_elos[row.home_team_name] = res['home_elo_post']
                self.current_elos[row.away_team_name] = res['away_elo_post']
            self.games.append(res)
        return pd.DataFrame(self.games)[upsert_load_columns]

    def rating_reset(self):
        """
        Regression towards the mean for team ratings.
        """
        team_names, elos = zip(*self.current_elos.items())
        diff_from_mean = np.array(elos) - self._mean_elo  # Default mean or actual list mean?
        elos -= diff_from_mean * (self._revert_percentage)
        self.current_elos = dict(zip(team_names, elos))


class EloGame:
    """
    EloGame class for simulating Elo ratings for a single game.

    Attributes:
        str_event_id (str): Event ID.
        season (int): Season of the game.
        date (pd.Timestamp): Date of the game.
        neutral_site (int): Flag indicating if the game is played at a neutral site.
        home_team_name (str): Name of the home team.
        home_team_score (int): Score of the home team.
        away_team_name (str): Name of the away team.
        away_team_score (int): Score of the away team.
        home_elo_pre (float): Initial Elo rating of the home team.
        home_elo_prob (float): Probability of the home team winning.
        home_elo_post (float): Final Elo rating of the home team.
        away_elo_pre (float): Initial Elo rating of the away team.
        away_elo_prob (float): Probability of the away team winning.
        away_elo_post (float): Final Elo rating of the away team.

    Methods:
        update_elo(k, hfa, width, allow_future): Update Elo ratings based on game outcome.
        sim(k, hfa, width, allow_future): Simulate Elo ratings for the game.

    """

    def __init__(
            self,
            str_event_id: str,
            season: int,
            date: pd.Timestamp,
            neutral_site: int,
            home_team_name: str,
            home_team_score: int,
            away_team_name: str,
            away_team_score: int,
            home_elo_pre: float = 1500.0,
            home_elo_prob: float = None,
            home_elo_post: float = None,
            away_elo_pre: float = 1500.0,
            away_elo_prob: float = None,
            away_elo_post: float = None,
    ):
        """
        Initialize EloGame.

        Args:
            str_event_id (str): Event ID.
            season (int): Season of the game.
            date (pd.Timestamp): Date of the game.
            neutral_site (int): Flag indicating if the game is played at a neutral site.
            home_team_name (str): Name of the home team.
            home_team_score (int): Score of the home team.
            away_team_name (str): Name of the away team.
            away_team_score (int): Score of the away team.
            home_elo_pre (float): Initial Elo rating of the home team.
            home_elo_prob (float): Probability of the home team winning.
            home_elo_post (float): Final Elo rating of the home team.
            away_elo_pre (float): Initial Elo rating of the away team.
            away_elo_prob (float): Probability of the away team winning.
            away_elo_post (float): Final Elo rating of the away team.
        """
        self.str_event_id = str_event_id
        self.season = season
        self.date = date
        self.neutral_site = neutral_site
        self.home_team_name = home_team_name
        self.home_team_score = home_team_score
        self.away_team_name = away_team_name
        self.away_team_score = away_team_score
        self.home_elo_pre = home_elo_pre
        self.home_elo_prob = home_elo_prob
        self.home_elo_post = home_elo_post
        self.away_elo_pre = away_elo_pre
        self.away_elo_prob = away_elo_prob
        self.away_elo_post = away_elo_post

    def update_elo(self, k=15, hfa=5, width=400, allow_future=False):
        """
        Update Elo ratings based on game outcome.

        Args:
            k (int): K Factor. Higher K = higher rating change.
            hfa (int): Home team advantage in Elo ratings.
            width (int): Lower and upper bounds of Elo ratings (mean_elo - width, mean_elo + width).
            allow_future (bool): Flag to include future events in simulation.

        Returns:
            Tuple: Tuple containing expected home and away shifts, new home Elo, and new away Elo.
        """
        try:
            # get expected home score
            elo_diff = self.home_elo_pre - self.away_elo_pre + (0 if self.neutral_site == 1 else hfa)
            expected_home_shift = 1.0 / (math.pow(10.0, (-elo_diff / width)) + 1.0)
            expected_away_shift = 1.0 / (math.pow(10.0, (elo_diff / width)) + 1.0)

            if is_pandas_none(self.away_team_score):
                self.away_team_score = None
            if is_pandas_none(self.home_team_score):
                self.home_team_score = None
            if self.away_team_score is None and self.home_team_score is None:
                if allow_future:
                    margin = expected_home_shift - expected_away_shift
                else:
                    return expected_home_shift, expected_away_shift, None, None
            else:
                margin = self.home_team_score - self.away_team_score

            if margin > 0:
                # shift of 1 for a win
                true_res = 1
            elif margin < 0:
                # shift of 0 for a loss
                true_res = 0
            else:
                # shift of 0.5 for a tie
                true_res = 0.5

            # Margin of victory multiplier calculation
            abs_margin = abs(margin)
            mult = math.log(max(abs_margin, 1) + 1.0) * (2.2 / (1.0 if true_res == 0.5 else ((elo_diff if true_res == 1.0 else -elo_diff) * 0.001 + 2.2)))

            # multiply difference of actual and expected score by k value and adjust home rating
            shift = (k * mult) * (true_res - expected_home_shift)
            new_home_elo = self.home_elo_pre + shift

            # repeat these steps for the away team
            # away shift is inverse of home shift
            new_away_elo = self.away_elo_pre - shift

            # return a tuple
            return expected_home_shift, expected_away_shift, new_home_elo, new_away_elo
        except ZeroDivisionError as e:
            print(e)
        except Exception as e:
            print(e)

    def sim(self, k=40, hfa=100, width=400, allow_future=False):
        """
        Simulate Elo ratings for the game.

        Args:
            k (int): K Factor. Higher K = higher rating change.
            hfa (int): Home team advantage in Elo ratings.
            width (int): Lower and upper bounds of Elo ratings (mean_elo - width, mean_elo + width).
            allow_future (bool): Flag to include future events in simulation.

        Returns:
            dict: Dictionary containing simulated Elo ratings for the game.
        """
        self.home_elo_prob, self.away_elo_prob, self.home_elo_post, self.away_elo_post = self.update_elo(k=k, hfa=hfa, width=width, allow_future=allow_future)
        return self.__dict__

def load_data(gender, sim_season):
    reg_compact_df = pd.read_csv(f"{DATA_PATH}/{gender}RegularSeasonCompactResults.csv")
    reg_compact_df['is_tourney_game'] = 0
    tourney_compact_df = pd.read_csv(f"{DATA_PATH}/{gender}NCAATourneyCompactResults.csv")
    tourney_compact_df['is_tourney_game'] = 1

    reg_detailed_df = pd.read_csv(f"{DATA_PATH}/{gender}RegularSeasonDetailedResults.csv")
    reg_detailed_df['is_tourney_game'] = 0
    tourney_detailed_df = pd.read_csv(f"{DATA_PATH}/{gender}NCAATourneyDetailedResults.csv")
    tourney_detailed_df['is_tourney_game'] = 1

    new_seeds = pd.read_csv(DATA_PATH + "2024_tourney_seeds.csv")
    new_seeds['Season'] = 2024
    new_seeds = new_seeds.drop(columns='Tournament')
    seeds_df = pd.concat([
        pd.read_csv(DATA_PATH + f"{gender}NCAATourneySeeds.csv"),
        new_seeds
    ], ignore_index=True)

    reg_df = pd.concat([
        reg_compact_df,
        reg_detailed_df
    ], ignore_index=True).drop_duplicates(subset=[
        'Season',
        'DayNum',
        'WTeamID',
        'LTeamID',
    ], keep='last').sort_values(['Season', 'DayNum'])

    del reg_compact_df
    del reg_detailed_df

    post_df = pd.concat([
        tourney_compact_df,
        tourney_detailed_df,
    ], ignore_index=True).drop_duplicates(subset=[
        'Season',
        'DayNum',
        'WTeamID',
        'LTeamID',
    ], keep='last').sort_values(['Season', 'DayNum'])
    del tourney_compact_df
    del tourney_detailed_df

    # Seeds Feature
    seeds_df['seed'] = seeds_df['Seed'].str.slice(start=1, stop=3).astype(int)
    post_df = pd.merge(post_df, seeds_df[['Season', 'TeamID', 'seed']].rename(columns={'seed': 'WSeed', 'TeamID': 'WTeamID'}), on=["Season", "WTeamID"], how="left")
    post_df = pd.merge(post_df, seeds_df[['Season', 'TeamID', 'seed']].rename(columns={'seed': 'LSeed', 'TeamID': 'LTeamID'}), on=["Season", "LTeamID"], how="left")
    del seeds_df

    # Avoid leakage for previous seasons
    reg_df = reg_df.loc[reg_df.Season<=sim_season].copy()
    post_df = post_df.loc[post_df.Season<sim_season].copy()

    ## Logic check:
    print('=' * 40)
    print('Experiment: Elo')
    print(f'Sim Season: {sim_season}')
    print(f'Building data for {sim_season} pre tourney state...')
    print(f'   - data up to {sim_season} regular season')
    print(f'   - data up to {sim_season-1} post season')
    print('=' * 40)

    df = pd.concat([
        reg_df,
        post_df
    ], ignore_index=True).drop_duplicates(subset=[
        'Season',
        'DayNum',
        'WTeamID',
        'LTeamID',
    ], keep='last').sort_values(['Season', 'DayNum'])

    del post_df

    # Conference Feature
    conf_df = pd.read_csv(f"{DATA_PATH}/{gender}TeamConferences.csv")
    df = pd.merge(df, conf_df[['Season', 'TeamID', 'ConfAbbrev']].rename(
        columns={'ConfAbbrev': 'WConference', 'TeamID': 'WTeamID'}), on=["Season", "WTeamID"], how="left")
    df = pd.merge(df, conf_df[['Season', 'TeamID', 'ConfAbbrev']].rename(
        columns={'ConfAbbrev': 'LConference', 'TeamID': 'LTeamID'}), on=["Season", "LTeamID"], how="left")
    del conf_df

    # Season and Team merge
    season_df = pd.read_csv(f"{DATA_PATH}/{gender}Seasons.csv")
    season_start_mapper = dict(zip(season_df.Season, pd.to_datetime(season_df.DayZero)))
    del season_df
    team_df = pd.read_csv(f"{DATA_PATH}/{gender}Teams.csv")
    team_name_mapper = dict(zip(team_df.TeamID, team_df.TeamName))
    del team_df

    # Date from DayNum
    df['date'] = df['Season'].map(season_start_mapper)
    df['date'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['DayNum'], unit='D')

    # Determine column name mapping
    df = df.rename(columns={
        'Season': 'season',
        'WLoc': 'neutral'
    })

    # Get all columns for winning team and losing team
    win_cols = [i for i in df.columns if i[0] == 'W']
    loss_cols = [i for i in df.columns if i[0] == 'L']

    # Make the dictionaries to pass to rename the columns
    cols = [i[1:].lower() for i in win_cols]
    home_cols = ['home_' + i[1:].lower() for i in win_cols]
    w_cols_dict = dict(zip(win_cols, home_cols))
    away_cols = ['away_' + i[1:].lower() for i in loss_cols]
    l_cols_dict = dict(zip(loss_cols, away_cols))

    df = df.rename(columns=w_cols_dict)
    df = df.rename(columns=l_cols_dict)

    # Converts from WTeam and LTeam to Home and Away
    for col in cols:
        temp = df.loc[df.neutral == 'A', f'away_{col}']
        df.loc[df.neutral == 'A', f'away_{col}'] = df.loc[df.neutral == 'A', f'home_{col}']
        df.loc[df.neutral == 'A', f'home_{col}'] = temp

    df.columns = [i.lower() for i in df.columns]
    df.neutral = df.neutral == 'N'

    df['away_team'] = df['away_teamid'].map(team_name_mapper)
    df['home_team'] = df['home_teamid'].map(team_name_mapper)

    df['game_id'] = df.away_teamid.astype(str) + "_" + df.home_teamid.astype(str) + "_" + df.date.dt.strftime('%Y%m%d')

    df['datetime'] = pd.to_datetime(df.date)
    df['has_ot'] = df.numot != 0
    df['is_conference_game'] = df['home_conference'] == df['away_conference']
    df = df.drop(columns=['numot', 'home_conference', 'away_conference'])

    ELO_COLS = [
        'season',
        'home_teamid',
        'home_score',
        'away_teamid',
        'away_score',
        'neutral',
        'game_id',
        'datetime'
    ]
    # Make Elo Rating
    elo_df = df[ELO_COLS].rename(columns={
        'game_id': 'str_event_id',
        'datetime': 'date',
        'home_teamid': 'home_team_name',
        'home_score': 'home_team_score',
        'away_teamid': 'away_team_name',
        'away_score': 'away_team_score',
        'neutral': 'neutral_site'
    })
    ELO_HYPERPARAMETERS = {
        'k': 30,
        'hfa': 100,
        'preloaded_elos': None
    }

    er = EloRunner(
        df=elo_df,
        allow_future=True,
        k=ELO_HYPERPARAMETERS['k'],
        mean_elo=1505,
        home_field_advantage=ELO_HYPERPARAMETERS['hfa'],
        width=800,
        preloaded_elos=None
    )
    elo_df = er.run_to_date()
    elo_df = elo_df.rename(columns={'home_team_name': 'home_team_id', 'away_team_name': 'away_team_id'})
    df = pd.merge(df, elo_df[['str_event_id', 'home_elo_pre', 'away_elo_pre', 'home_elo_prob', 'away_elo_prob','home_team_id','away_team_id','neutral_site','home_elo_post','away_elo_post']].rename(
        columns={'str_event_id': 'game_id'}), on=['game_id'], how='left')
    df = df.sort_values('date')
    return df

def generate_submission_file(season):
    submission = []
    for gender in ['M','W']:
        subs = []
        teams=pd.read_csv(DATA_PATH + f"{gender}Teams.csv")
        teamids = list(teams.TeamID.unique())
        teamids = sorted(teamids)
        for low_team_id in teamids:
            for high_team_id in teamids:
                if low_team_id < high_team_id:
                    subs.append(f"{season}_{low_team_id}_{high_team_id}")
        sub = pd.DataFrame({'ID':subs})
        sub['Tournament'] = gender
        submission.append(sub)
    return pd.concat(submission, ignore_index=True)

def preprocessing(sim_season, elo):
    sub = generate_submission_file(sim_season)
    new_seeds = pd.read_csv(DATA_PATH + "2024_tourney_seeds.csv")
    new_seeds['Season'] = 2024
    new_seeds = new_seeds.drop(columns='Tournament')
    seeds = pd.concat([
        pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
        pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
        new_seeds
    ], ignore_index=True)
    seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    seeds_T1 = seeds[['Season', 'TeamID', 'seed']].copy()
    seeds_T2 = seeds[['Season', 'TeamID', 'seed']].copy()
    seeds_T1.columns = ['Season', 'T1_TeamID', 'T1_seed']
    seeds_T2.columns = ['Season', 'T2_TeamID', 'T2_seed']

    sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
    sub["T1_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
    sub["T2_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[2]))
    sub['T1_TeamElo'] = sub['T1_TeamID'].replace(elo)
    sub['T2_TeamElo'] = sub['T2_TeamID'].replace(elo)
    sub = pd.merge(sub, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
    sub = pd.merge(sub, seeds_T2, on=['Season', 'T2_TeamID'], how='left')
    return sub

def predict(sub):
    games = []
    for idx,row in sub.iterrows():

        dict_row = {
            'str_event_id': row['ID'],
            'season': 1,
            'date': pd.Timestamp('1900-01-01'),
            'neutral_site': 1,
            'home_team_name': row['T1_TeamID'],
            'home_team_score': None,
            'away_team_name': row['T2_TeamID'],
            'away_team_score': None,
            'home_elo_pre': row['T1_TeamElo'],
            'home_elo_prob': None,
            'home_elo_post': None,
            'away_elo_pre': row['T2_TeamElo'],
            'away_elo_prob': None,
            'away_elo_post': None,
        }
        elo_game = EloGame(**dict_row)
        res = elo_game.sim(
            k=30,
            hfa=100,
            width=800,
            allow_future=True
        )
        # update metadata record
        dict_row['home_elo_prob'] = res['home_elo_prob']
        dict_row['home_elo_post'] = res['home_elo_post']
        dict_row['away_elo_prob'] = res['away_elo_prob']
        dict_row['away_elo_post'] = res['away_elo_post']
        games.append({
            'ID':row['ID'],
            'Pred':dict_row['home_elo_prob'],
            'Tournament':row['Tournament'],
            'Season':row['Season'],
            'T1_TeamID':row['T1_TeamID'],
            'T2_TeamID':row['T2_TeamID'],
            'T1_seed':row['T1_seed'],
            'T2_seed':row['T2_seed'],
        })
    return pd.DataFrame(games)

def get_latest_elo(df):
    folded_elo_df = df_rename_fold(df[['game_id', 'season', 'date', 'neutral_site',
                                             'home_team_id', 'away_team_id', 'home_elo_pre',
                                            'away_elo_pre', 'home_elo_post', 'away_elo_post']], 'away_',
                                   'home_').sort_values('date')
    current_ratings_df = folded_elo_df.groupby('team_id').nth(-1).sort_values(['elo_post'], ascending=False)
    current_ratings_df = current_ratings_df.drop(columns=[ 'elo_pre']).rename(columns={'elo_post': 'elo_rating', 'date': 'lastupdated'})
    current_ratings_df = current_ratings_df.loc[current_ratings_df.season >= current_ratings_df.season.max() - 1]
    current_ratings_df['rank'] = [i + 1 for i in range(current_ratings_df.shape[0])]
    current_ratings_df = current_ratings_df[['team_id', 'rank', 'elo_rating', 'lastupdated']].copy()
    ids = list(map(int, list(current_ratings_df.team_id.values)))
    return dict(zip(ids,list(current_ratings_df.elo_rating.values)))

def put_elo_ratings(df, path):
    elo = get_latest_elo(df)
    with open(path, 'w') as json_file:
        json.dump(elo, json_file, indent=2)

def runner(sim_season):
    '''
    Generate predictions for march madness following previously successful algorithm and data processing. Preprocess the data, fit historical data to an XGboost model, predict game results and save them to a csv
    :param sim_season: The season to generate predictions for
    :param repeat_cv: Models to build and average across (boosted, bagging)
    :param upset_overrides: Force score that no upset will happen for the certain list of matchups that an upset has never happened for
    :return:
    '''
    out_columns = ['ID','Pred','Tournament','Season','T1_TeamID','T2_TeamID']
    m_data= load_data('M', sim_season)
    w_data= load_data('W', sim_season)
    df = pd.concat([m_data,w_data],axis=0)
    elo = get_latest_elo(df)
    sub = preprocessing(sim_season, elo)
    sub = predict(sub)
    sub = sub.loc[((sub.T1_seed.notnull()) & (sub.T2_seed.notnull()))]

    path = f'./output/elo/{sim_season}/'
    os.makedirs(path,exist_ok=True)
    put_elo_ratings(df,f'{path}latest_elo.json')
    sub[out_columns].to_csv(f"{path}predictions.csv", index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MMLM Elo Prediction')
    parser.add_argument('sim_season', type=int, help='The season to generate predictions for')
    args = parser.parse_args()
    runner(args.sim_season)