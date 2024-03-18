import argparse
import datetime
import json

from sklearn.model_selection import cross_val_score, train_test_split
from scipy.stats import linregress
from tqdm import tqdm

import glob
import lightgbm as lgb
import numpy as np
import optuna as op
import os
import pandas as pd
import argparse
import datetime

import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
op.logging.set_verbosity(op.logging.WARNING)

from utils import DATA_PATH

def prepare_data(df):
    dfswap = df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'WLoc', 'NumOT',
                 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
                 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]

    dfswap.loc[df['WLoc'] == 'H', 'WLoc'] = 'A'
    dfswap.loc[df['WLoc'] == 'A', 'WLoc'] = 'H'
    df.columns.values[6] = 'location'
    dfswap.columns.values[6] = 'location'

    df.columns = [x.replace('W', 'T1_').replace('L', 'T2_') for x in list(df.columns)]
    dfswap.columns = [x.replace('L', 'T1_').replace('W', 'T2_') for x in list(dfswap.columns)]

    output = pd.concat([df, dfswap]).reset_index(drop=True)
    output.loc[output.location == 'N', 'location'] = '0'
    output.loc[output.location == 'H', 'location'] = '1'
    output.loc[output.location == 'A', 'location'] = '-1'
    output.location = output.location.astype(int)

    output['PointDiff'] = output['T1_Score'] - output['T2_Score']
    output['PointDiffAgainst'] = output['T2_Score'] - output['T1_Score']
    return output

def build_season_statistics(regular_data):
    # Handle Avg Boxscore calculations
    boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF',
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',
        'PointDiff','PointDiffAgainst']

    funcs = [np.mean]

    season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
    season_statistics.columns = ['_'.join(col).strip() for col in season_statistics.columns.values]
    season_statistics_T1 = season_statistics.copy()
    season_statistics_T2 = season_statistics.copy()

    # Column renaming
    season_statistics_T1.columns = ["T1_" + x.replace("T1_", "off_").replace("T2_", "def_") for x in list(season_statistics_T1.columns)]
    season_statistics_T2.columns = ["T2_" + x.replace("T1_", "off_").replace("T2_", "def_") for x in list(season_statistics_T2.columns)]
    season_statistics_T1.columns.values[0] = "Season"
    season_statistics_T2.columns.values[0] = "Season"
    season_statistics_T1.columns.values[1] = "T1_TeamID"
    season_statistics_T2.columns.values[1] = "T2_TeamID"
    return season_statistics_T1, season_statistics_T2

def build_score_attrs(regular_data):
    season_score_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[['T1_Score', 'T2_Score']].agg(['mean','min', 'max', 'std']).reset_index()
    season_score_statistics.columns = ['_'.join(col).strip() for col in season_score_statistics.columns.values]
    season_score_statistics_T1 = season_score_statistics.copy()
    season_score_statistics_T2 = season_score_statistics.copy()

    # Column renaming
    season_score_statistics_T1.columns = ["T1_" + x.replace("T1_", "off_").replace("T2_", "def_") for x in list(season_score_statistics_T1.columns)]
    season_score_statistics_T2.columns = ["T2_" + x.replace("T1_", "off_").replace("T2_", "def_") for x in list(season_score_statistics_T2.columns)]
    season_score_statistics_T1.columns.values[0] = "Season"
    season_score_statistics_T2.columns.values[0] = "Season"
    season_score_statistics_T1.columns.values[1] = "T1_TeamID"
    season_score_statistics_T2.columns.values[1] = "T2_TeamID"
    return season_score_statistics_T1, season_score_statistics_T2

def build_win_perc(regular_data):
    last14days_stats_T1 = regular_data.reset_index(drop=True)
    last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff'] > 0, 1, 0)
    last14days_stats_T1 = last14days_stats_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio')
    last14days_stats_T2 = regular_data.reset_index(drop=True)
    last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff'] < 0, 1, 0)
    last14days_stats_T2 = last14days_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio')
    return last14days_stats_T1, last14days_stats_T2

def build_last_14_stats(regular_data):
    # Adds last 14 day stats
    last14days_stats_T1 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff'] > 0, 1, 0)
    last14days_stats_T1 = last14days_stats_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')
    last14days_stats_T2 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff'] < 0, 1, 0)
    last14days_stats_T2 = last14days_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')
    return last14days_stats_T1, last14days_stats_T2

def build_seeds(seeds):
    # Adds Seeds
    seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
    seeds_T1 = seeds[['Season', 'TeamID', 'seed']].copy()
    seeds_T2 = seeds[['Season', 'TeamID', 'seed']].copy()
    seeds_T1.columns = ['Season', 'T1_TeamID', 'T1_seed']
    seeds_T2.columns = ['Season', 'T2_TeamID', 'T2_seed']
    return seeds_T1, seeds_T2

def team_quality(season, regular_season_effects):
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula,
                              data=regular_season_effects.loc[regular_season_effects.Season == season, :],
                              family=sm.families.Binomial()).fit()

    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID', 'quality']
    quality['Season'] = season
    # quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality

def build_quality(regular_season_effects, seeds, sim_season):
    # Create effects dataframe to feed to quality model

    regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
    regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
    regular_season_effects['win'] = np.where(regular_season_effects['PointDiff'] > 0, 1, 0)
    march_madness = pd.merge(seeds[['Season', 'TeamID']], seeds[['Season', 'TeamID']], on='Season')
    march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
    march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
    march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
    regular_season_effects = pd.merge(regular_season_effects, march_madness, on=['Season', 'T1_TeamID', 'T2_TeamID'])

    seasons = [season for season in list(range(2003, sim_season + 1)) if season != 2020]
    glm_quality = pd.concat([team_quality(season, regular_season_effects) for season in seasons]).reset_index(drop=True)

    glm_quality_T1 = glm_quality.copy()
    glm_quality_T2 = glm_quality.copy()
    glm_quality_T1.columns = ['T1_TeamID', 'T1_quality', 'Season']
    glm_quality_T2.columns = ['T2_TeamID', 'T2_quality', 'Season']
    return glm_quality_T1, glm_quality_T2

def generate_submission_file(season):
    submission = []
    for gender in ['M','W']:
        subs = []
        teams=pd.read_csv(DATA_PATH + f"{gender}Teams.csv")
        teamids = list(teams.TeamID.unique())
        teamids = sorted(teamids)
        for low_team_id in teamids:
            for high_team_id in teamids:
                subs.append(f"{season}_{low_team_id}_{high_team_id}")
        sub = pd.DataFrame({'ID':subs})
        sub['Pred'] = 0.5
        sub['Tournament'] = gender
        submission.append(sub)
    return pd.concat(submission, ignore_index=True)

def make_features(
        sim_season,
        season_statistics_T1,
        season_statistics_T2,
        season_score_statistics_T1,
        season_score_statistics_T2,
        last14days_stats_T1,
        last14days_stats_T2,
        wp_stats_T1,
        wp_stats_T2,
        seeds_T1,
        seeds_T2,
        glm_quality_T1=None,
        glm_quality_T2=None,
        tourney_data = None,
):
    # Merge regular season stats to tourney data
    if tourney_data is not None:
        tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score']]
        is_test = False
    else:
        tourney_data = generate_submission_file(sim_season)
        tourney_data['Season'] = tourney_data['ID'].apply(lambda x: int(x.split('_')[0]))
        tourney_data["T1_TeamID"] = tourney_data['ID'].apply(lambda x: int(x.split('_')[1]))
        tourney_data["T2_TeamID"] = tourney_data['ID'].apply(lambda x: int(x.split('_')[2]))
        is_test = True
    tourney_data = pd.merge(tourney_data, season_statistics_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, season_statistics_T2, on=['Season', 'T2_TeamID'], how='left')

    tourney_data = pd.merge(tourney_data, season_score_statistics_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, season_score_statistics_T2, on=['Season', 'T2_TeamID'], how='left')

    # Merge last 14 day stats to tourney data
    tourney_data = pd.merge(tourney_data, last14days_stats_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, last14days_stats_T2, on=['Season', 'T2_TeamID'], how='left')

    # Merge win percentage stats to tourney data
    tourney_data = pd.merge(tourney_data, wp_stats_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, wp_stats_T2, on=['Season', 'T2_TeamID'], how='left')

    # Merge seeds to tourney data
    tourney_data = pd.merge(tourney_data, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, seeds_T2, on=['Season', 'T2_TeamID'], how='left')

    # Merge quality to tourney data (adds optional case for adding this feature cause it takes a bit to compute)
    if glm_quality_T1 is not None and glm_quality_T2 is not None:
        tourney_data = pd.merge(tourney_data, glm_quality_T1, on=['Season', 'T1_TeamID'], how='left')
        tourney_data = pd.merge(tourney_data, glm_quality_T2, on=['Season', 'T2_TeamID'], how='left')
    else:
        tourney_data['T1_quality'] = 0.001
        tourney_data['T2_quality'] = 0.001

    # Create seed diff feature and drop seeds
    tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]
    if not is_test:
        # Create the tourney point diff incase we want to use this as the target again
        tourney_data['PointDiff'] = tourney_data['T1_Score'] - tourney_data['T2_Score']
        tourney_data = tourney_data.sort_values(['Season', 'DayNum'])

    return tourney_data


def preprocessing(sim_season):
    ## Import data
    tourney_results = pd.concat([
        pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
        pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
    ], ignore_index=True)

    new_seeds = pd.read_csv(DATA_PATH + "2024_tourney_seeds.csv")
    new_seeds['Season'] = 2024
    new_seeds = new_seeds.drop(columns='Tournament')
    seeds = pd.concat([
        pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
        pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
        new_seeds
    ], ignore_index=True)

    regular_results = pd.concat([
        pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv"),
        pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv"),
    ], ignore_index=True)

    regular_data = prepare_data(regular_results)
    tourney_data = prepare_data(tourney_results)

    # Avoid leakage for previous seasons
    regular_data = regular_data.loc[regular_data.Season <= sim_season].copy()
    tourney_data = tourney_data.loc[tourney_data.Season < sim_season].copy()

    ## Logic check:
    print('=' * 40)
    print('Experiment: madness-2024-sub')
    print(f'Sim Season: {sim_season}')
    print(f'Building data for {sim_season} pre tourney state...')
    print(f'   - data up to {regular_data.Season.max()} regular season')
    print(f'   - data up to {tourney_data.Season.max()} post season')
    print(f'   - Predicting {sim_season} post season')
    print('=' * 40)

    # Handle Avg Boxscore calculations
    season_statistics_T1, season_statistics_T2 = build_season_statistics(regular_data)
    season_score_statistics_T1, season_score_statistics_T2 = build_score_attrs(regular_data)

    last14days_stats_T1, last14days_stats_T2 = build_last_14_stats(regular_data)

    wp_stats_T1, wp_stats_T2 = build_win_perc(regular_data)

    seeds_T1, seeds_T2 = build_seeds(seeds)

    regular_season_effects = regular_data[['Season', 'T1_TeamID', 'T2_TeamID', 'PointDiff']].copy()
    glm_quality_T1, glm_quality_T2 = build_quality(regular_season_effects, seeds, sim_season)
    #glm_quality_T1, glm_quality_T2 = None, None
    tourney_data = make_features(
        sim_season,
        season_statistics_T1,
        season_statistics_T2,
        season_score_statistics_T1,
        season_score_statistics_T2,
        last14days_stats_T1,
        last14days_stats_T2,
        wp_stats_T1,
        wp_stats_T2,
        seeds_T1,
        seeds_T2,
        glm_quality_T1,
        glm_quality_T2,
        tourney_data=tourney_data,
    )
    sub = make_features(
        sim_season,
        season_statistics_T1,
        season_statistics_T2,
        season_score_statistics_T1,
        season_score_statistics_T2,
        last14days_stats_T1,
        last14days_stats_T2,
        wp_stats_T1,
        wp_stats_T2,
        seeds_T1,
        seeds_T2,
        glm_quality_T1,
        glm_quality_T2,
        tourney_data=None,
    )
    stat_cols = [i for i in tourney_data.columns for j in ['_min', '_max', '_std', '_mean'] if j in i]
    features = [
        'Season',
        #'DayNum',
        'T1_TeamID',
        #'T1_Score',
        'T2_TeamID',
        #'T2_Score',
        'T1_win_ratio_14d',
        'T2_win_ratio_14d',
        'T1_win_ratio',
        'T2_win_ratio',
        'T1_quality',
        'T2_quality',
        'T1_seed',
        'T2_seed',
        'Seed_diff'
    ] + stat_cols
    return tourney_data, sub, features

def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x =  preds-labels
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess

def objective(trial, X, y):
    params = {
        'eval_metric': 'mae',
        'booster': 'gbtree',
        'alpha': trial.suggest_float('alpha', 1e-5, 0.1, log=True),
        'lambda': trial.suggest_float('lambda', 1e-5, 0.1, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4,0.1, log=True),
        'gamma': trial.suggest_int('gamma', 0, 30),
        'subsample': trial.suggest_float('subsample', 0.1, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'num_parallel_tree': trial.suggest_int('num_parallel_tree', 5, 31),
        'verbosity': 0
    }
    dtrain = xgb.DMatrix(X, label=y)
    pruning_callback = op.integration.XGBoostPruningCallback(trial, "test-mae")
    history = xgb.cv(
        params,
        dtrain,
        nfold=5,
        obj=cauchyobj,
        early_stopping_rounds=25,
        verbose_eval=500,
        num_boost_round=4000,
        callbacks=[pruning_callback]
    )
    return history["test-mae-mean"].values[-1]


def study(X, y, n_trials):
    pruner = op.pruners.MedianPruner(n_warmup_steps=5)
    s = op.create_study(pruner=pruner,direction='minimize')
    s.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
    return s

# Define a function to create the ID and adjust prediction values
def create_id_and_adjust_preds(row):
    # Sort the TeamIDs and join them with underscores to create the ID
    team_ids = sorted([row['T1_TeamID'], row['T2_TeamID']])
    team1_id, team2_id = team_ids
    id_str = f"{row['Season']}_{team1_id}_{team2_id}"

    # Check if the TeamIDs need to be flipped
    if team1_id != row['T1_TeamID']:
        # Swap the prediction values if the TeamIDs are flipped
        t1_pred, t2_pred = row['T2Pred'], row['T1Pred']
    else:
        # Use the original prediction values
        t1_pred, t2_pred = row['T1Pred'], row['T2Pred']

    return pd.Series([id_str, t1_pred, t2_pred], index=['ID', 'T1Pred', 'T2Pred'])

def flip_and_shift_pred(meta, pred):
    actual = meta.copy()
    meta['T1Pred'] = pred
    meta['T2Pred'] = None
    meta[['ID', 'T1Pred', 'T2Pred']] = meta.apply(create_id_and_adjust_preds, axis=1)
    t1_meta = meta[['ID', 'T1Pred']].loc[meta.T1Pred.notnull()].copy()
    t2_meta = meta[['ID', 'T2Pred']].loc[meta.T2Pred.notnull()].copy()

    pred_df = pd.merge(t1_meta, t2_meta, on=['ID'])
    pred_df['Pred_Diff'] = pred_df['T1Pred'] - pred_df['T2Pred']
    pred_df['Pred_Total'] = pred_df['T1Pred'] + pred_df['T2Pred']
    pred_df['Season'] = pred_df['ID'].apply(lambda x: int(x.split('_')[0]))
    pred_df["T1_TeamID"] = pred_df['ID'].apply(lambda x: int(x.split('_')[1]))
    pred_df["T2_TeamID"] = pred_df['ID'].apply(lambda x: int(x.split('_')[2]))
    reg = pred_df.copy()
    swap = pred_df.copy()
    swap['Pred_Diff'] = -swap['Pred_Diff']
    swap.columns = ['ID', 'T2Pred', 'T1Pred', 'Pred_Diff', 'Pred_Total', 'Season', 'T2_TeamID', 'T1_TeamID']
    meta = pd.concat([
        reg,
        swap
    ])
    meta = pd.merge(actual, meta, on=['Season', 'T1_TeamID', 'T2_TeamID'], how='left')
    return meta

def modeling(sim_season,tourney_data, features, repeat_cv = 3):
    target = [
        'T1_Score'
    ]

    tourney_data = tourney_data.reset_index(drop=True)
    meta = tourney_data[['Season', 'T1_TeamID', 'T1_Score',  'T2_TeamID', 'T2_Score']].copy()
    X, y = tourney_data[features].copy(), tourney_data[target].copy()
    X = X.values
    y = y.values

    avg = tourney_data[target].values.mean()
    std = tourney_data[target].values.std()

    upper_bound = int(avg + std * 3)
    lower_bound = int(avg - std * 3)
    print(f'Target Values from {round(lower_bound)} - {round(upper_bound)}')

    ### Add HPO Option Here if it hasnt been run before
    path = f'./output/madness-2024-sub/{sim_season}/'
    os.makedirs(path, exist_ok=True)
    try:
        with open(path+'hyperparams.json', 'r') as json_file:
            param = json.load(json_file)
    except Exception as e:
        print(f"NO HPO FOUND FOR {sim_season}. RUNNING NOW...")
        param = study(X, y, n_trials=5)
        param = param.best_trial.params
        with open(path+'hyperparams.json', 'w') as json_file:
            json.dump(param, json_file, indent=2)
        path = f'./output/madness-2024-sub-proba/{sim_season}/'
        with open(path+'hyperparams.json', 'w') as json_file:
            json.dump(param, json_file, indent=2)

    ### Add HPO Option Here if it hasnt been run before
    path = f'./output/madness-2024-sub-proba/{sim_season}/'
    os.makedirs(path, exist_ok=True)
    try:
        with open(path+'hyperparams.json', 'r') as json_file:
            param = json.load(json_file)
    except Exception as e:
        print(f"NO HPO FOUND FOR {sim_season}. RUNNING NOW...")
        param = study(X, y, n_trials=5)
        param = param.best_trial.params
        with open(path+'hyperparams.json', 'w') as json_file:
            json.dump(param, json_file, indent=2)
        path = f'./output/madness-2024-sub/{sim_season}/'
        with open(path+'hyperparams.json', 'w') as json_file:
            json.dump(param, json_file, indent=2)

    # param['objective'] = 'reg:linear'
    param['eval_metric'] = 'mae'
    param['booster'] = 'gbtree'
    param['silent'] = 1

    dtrain = xgb.DMatrix(X, label=y)

    xgb_cv = []
    for i in range(repeat_cv):
        print(f"Fold repeater {i}")
        xgb_cv.append(
            xgb.cv(
                params=param,
                dtrain=dtrain,
                obj=cauchyobj,
                num_boost_round=3000,
                folds=KFold(n_splits=5, shuffle=True, random_state=i),
                early_stopping_rounds=25,
                verbose_eval=50
            )
        )

    iteration_counts = [np.argmin(x['test-mae-mean'].values) for x in xgb_cv]
    val_mae = [np.min(x['test-mae-mean'].values) for x in xgb_cv]
    print(f"Avg CV validation MAE: {np.array(val_mae).mean()}")

    oof_preds = []
    for i in range(repeat_cv):
        print(f"Fold repeater {i}")
        preds = y.ravel().copy().astype(float)
        kfold = KFold(n_splits=5, shuffle=True, random_state=i)
        for train_index, val_index in kfold.split(X, y):
            dtrain_i = xgb.DMatrix(X[train_index], label=y[train_index])
            dval_i = xgb.DMatrix(X[val_index], label=y[val_index])
            model = xgb.train(
                params=param,
                dtrain=dtrain_i,
                obj=cauchyobj,
                num_boost_round=iteration_counts[i],
                verbose_eval=50
            )
            preds[val_index] = model.predict(dval_i)
        oof_preds.append(np.clip(preds, lower_bound, upper_bound))
    all_preds_avg = np.array(oof_preds).mean(axis=0)  # Mean Across CV iterations
    pred_avg = all_preds_avg.mean()  # Mean across Mean CV iterations
    pred_std = all_preds_avg.std()
    pred_upper_bound = int(pred_avg + pred_std * 3)
    pred_lower_bound = int(pred_avg - pred_std * 3)
    print(f'Train Pred Values from {round(pred_lower_bound)} - {round(pred_upper_bound)}')

    val_cv = []
    spline_model = []

    for i in range(repeat_cv):
        pred_data_df = flip_and_shift_pred(meta, oof_preds[i])
        y_diff = pred_data_df.T1_Score - pred_data_df.T2_Score
        pred = np.clip(pred_data_df.Pred_Diff.values, -30, 30)
        dat = list(zip(pred, np.where(y_diff > 0, 1, 0)))
        dat = sorted(dat, key=lambda x: x[0])
        datdict = {}
        for k in range(len(dat)):
            datdict[dat[k][0]] = dat[k][1]
        spline_model.append(UnivariateSpline(list(datdict.keys()), list(datdict.values())))
        spline_fit = spline_model[i](oof_preds[i])
        spline_fit = np.clip(spline_fit, 0.025, 0.975)
        spline_fit[(tourney_data.T1_seed == 1) & (tourney_data.T2_seed == 16) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
        spline_fit[(tourney_data.T1_seed == 2) & (tourney_data.T2_seed == 15) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
        spline_fit[(tourney_data.T1_seed == 3) & (tourney_data.T2_seed == 14) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
        spline_fit[(tourney_data.T1_seed == 4) & (tourney_data.T2_seed == 13) & (tourney_data.T1_Score > tourney_data.T2_Score)] = 1.0
        spline_fit[(tourney_data.T1_seed == 16) & (tourney_data.T2_seed == 1) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
        spline_fit[(tourney_data.T1_seed == 15) & (tourney_data.T2_seed == 2) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
        spline_fit[(tourney_data.T1_seed == 14) & (tourney_data.T2_seed == 3) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0
        spline_fit[(tourney_data.T1_seed == 13) & (tourney_data.T2_seed == 4) & (tourney_data.T1_Score < tourney_data.T2_Score)] = 0.0

        val_cv.append(pd.DataFrame({"y": np.where(y_diff > 0, 1, 0), "pred": spline_fit, "season": tourney_data.Season}))
        print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y_diff > 0, 1, 0), spline_fit)}")

    val_cv = pd.concat(val_cv)
    print(val_cv.groupby('season').apply(lambda x: log_loss(x.y, x.pred)))

    sub_models = []
    for i in range(repeat_cv):
        print(f"Fold repeater {i}")
        sub_models.append(
            xgb.train(
                params=param,
                dtrain=dtrain,
                obj=cauchyobj,
                num_boost_round=int(iteration_counts[i] * 1.05),
                verbose_eval=50
            )
        )

    importances = []
    for i in range(repeat_cv):
        booster = sub_models[i]
        feature_importance = booster.get_score(importance_type='weight')
        mapped_feature_importance = {features[int(key[1:])]: value for key, value in feature_importance.items()}
        importances.append(mapped_feature_importance)

    df_importances = pd.DataFrame(importances).mean(axis=0).reset_index().rename(columns={0: 'score', 'index': 'col'}).sort_values('score', ascending=False)
    print('Avg Feature Importance for Trained Model: ')
    print(df_importances)
    print()
    print(f'Potential Features to drop: {df_importances.col.values[-10:]}')
    return {
        'models': sub_models,
        'spline_model': spline_model,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

def predict(model_obj, sub, features, repeat_cv = 3, upset_overrides=False):
    sub_models = model_obj['models']
    lower_bound = model_obj['lower_bound']
    upper_bound = model_obj['upper_bound']
    spline_model = model_obj['spline_model']

    Xsub = sub.loc[sub.Seed_diff.notnull()].copy()[features].values
    dtest = xgb.DMatrix(Xsub)
    sub_preds = []
    sub_metas = []

    for i in range(repeat_cv):
        point_pred = np.clip(sub_models[i].predict(dtest), lower_bound, upper_bound)
        meta = sub.loc[sub.Seed_diff.notnull()].copy()[['Season', 'T1_TeamID', 'T2_TeamID', 'Tournament']]
        meta['T1_Score'] = None
        meta['T2_Score'] = None
        meta_out = flip_and_shift_pred(meta, point_pred)
        meta_out = meta_out.loc[meta_out.T1_TeamID < meta_out.T2_TeamID].copy()
        meta_out['Tournament'] = np.where(meta_out.T1_TeamID < 2000, 'M', 'W')
        pred = np.clip(spline_model[i](np.clip(meta_out.Pred_Diff.values, -30, 30)), 0.025, 0.975)
        sub_preds.append(pred)
        meta_out['Pred'] = pred
        sub_metas.append(meta_out)
    temp = sub[['ID','T1_seed','T2_seed']].copy()
    sub = pd.concat(sub_metas)
    sub = sub.groupby(['ID', 'Tournament']).mean().reset_index()
    sub['Season'] = sub['Season'].astype(int)
    sub['T1_TeamID'] = sub['T1_TeamID'].astype(int)
    sub['T2_TeamID'] = sub['T2_TeamID'].astype(int)
    sub = pd.merge(sub, temp, on=['ID'], how='left')

    if upset_overrides:
        sub.loc[(sub.T1_seed == 1) & (sub.T2_seed == 16), 'Pred'] = 1.0
        sub.loc[(sub.T1_seed == 2) & (sub.T2_seed == 15), 'Pred'] = 1.0
        sub.loc[(sub.T1_seed == 3) & (sub.T2_seed == 14), 'Pred'] = 1.0
        sub.loc[(sub.T1_seed == 4) & (sub.T2_seed == 13), 'Pred'] = 1.0
        sub.loc[(sub.T1_seed == 16) & (sub.T2_seed == 1), 'Pred'] = 0.0
        sub.loc[(sub.T1_seed == 15) & (sub.T2_seed == 2), 'Pred'] = 0.0
        sub.loc[(sub.T1_seed == 14) & (sub.T2_seed == 3), 'Pred'] = 0.0
        sub.loc[(sub.T1_seed == 13) & (sub.T2_seed == 4), 'Pred'] = 0.0
    return sub


def runner(sim_season, repeat_cv=3, upset_overrides=False):
    '''
    Generate predictions for march madness following previously successful algorithm and data processing. Preprocess the data, fit historical data to an XGboost model, predict game results and save them to a csv
    :param sim_season: The season to generate predictions for
    :param repeat_cv: Models to build and average across (boosted, bagging)
    :param upset_overrides: Force score that no upset will happen for the certain list of matchups that an upset has never happened for
    :return:
    '''
    tourney_data, sub, features = preprocessing(sim_season)
    model_obj = modeling(sim_season, tourney_data, features, repeat_cv=repeat_cv)
    sub = predict(model_obj, sub, features, repeat_cv=repeat_cv, upset_overrides=upset_overrides)
    path = f'./output/madness-2024-sub/{sim_season}/'
    os.makedirs(path, exist_ok=True)
    sub.to_csv(f"{path}predictions.csv", index=None)
    path = f'./output/madness-2024-sub-proba/{sim_season}/'
    os.makedirs(path, exist_ok=True)
    sub.to_csv(f"{path}predictions.csv", index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='2024 Submission March Madness')
    parser.add_argument('sim_season', type=int, help='The season to generate predictions for')
    parser.add_argument('--repeat_cv', type=int, default=3, help='Models to build and average across (boosted, bagging)')
    parser.add_argument('--upset_overrides', action='store_true', help='Force score that no upset will happen for the certain list of matchups that an upset has never happened for')

    args = parser.parse_args()
    runner(args.sim_season, args.repeat_cv, args.upset_overrides)







