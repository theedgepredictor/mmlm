import argparse
import datetime

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.interpolate import UnivariateSpline
import statsmodels.api as sm
import collections

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
        sub['Pred'] = 0.5
        sub['Tournament'] = gender
        submission.append(sub)
    return pd.concat(submission, ignore_index=True)

def preprocessing(sim_season=datetime.datetime.now().year):
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
    regular_data = regular_data.loc[regular_data.Season<=sim_season].copy()
    tourney_data = tourney_data.loc[tourney_data.Season<sim_season].copy()

    ## Logic check:
    print('=' * 40)
    print('Experiment: madness-2023-1st')
    print(f'Sim Season: {sim_season}')
    print(f'Building data for {sim_season} pre tourney state...')
    print(f'   - data up to {sim_season} regular season')
    print(f'   - data up to {sim_season-1} post season')
    print(f'   - Predicting {sim_season} post season')
    print('=' * 40)

    # Handle Avg Boxscore calculations
    boxscore_cols = [
        'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_OR', 'T1_Ast', 'T1_TO', 'T1_Stl', 'T1_PF',
        'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_OR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk',
        'PointDiff']

    funcs = [np.mean]

    season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg(funcs).reset_index()
    season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]
    season_statistics_T1 = season_statistics.copy()
    season_statistics_T2 = season_statistics.copy()

    # Column renaming
    season_statistics_T1.columns = ["T1_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(season_statistics_T1.columns)]
    season_statistics_T2.columns = ["T2_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in list(season_statistics_T2.columns)]
    season_statistics_T1.columns.values[0] = "Season"
    season_statistics_T2.columns.values[0] = "Season"

    # Merge regular season stats to tourney data
    tourney_data = tourney_data[['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score']]
    tourney_data = pd.merge(tourney_data, season_statistics_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, season_statistics_T2, on=['Season', 'T2_TeamID'], how='left')

    # Adds last 14 day stats
    last14days_stats_T1 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff'] > 0, 1, 0)
    last14days_stats_T1 = last14days_stats_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(name='T1_win_ratio_14d')
    last14days_stats_T2 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff'] < 0, 1, 0)
    last14days_stats_T2 = last14days_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(name='T2_win_ratio_14d')

    # Merge last 14 day stats to tourney data
    tourney_data = pd.merge(tourney_data, last14days_stats_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, last14days_stats_T2, on=['Season', 'T2_TeamID'], how='left')

    # Create effects dataframe to feed to quality model
    regular_season_effects = regular_data[['Season', 'T1_TeamID', 'T2_TeamID', 'PointDiff']].copy()
    regular_season_effects['T1_TeamID'] = regular_season_effects['T1_TeamID'].astype(str)
    regular_season_effects['T2_TeamID'] = regular_season_effects['T2_TeamID'].astype(str)
    regular_season_effects['win'] = np.where(regular_season_effects['PointDiff'] > 0, 1, 0)
    march_madness = pd.merge(seeds[['Season', 'TeamID']], seeds[['Season', 'TeamID']], on='Season')
    march_madness.columns = ['Season', 'T1_TeamID', 'T2_TeamID']
    march_madness.T1_TeamID = march_madness.T1_TeamID.astype(str)
    march_madness.T2_TeamID = march_madness.T2_TeamID.astype(str)
    regular_season_effects = pd.merge(regular_season_effects, march_madness, on=['Season', 'T1_TeamID', 'T2_TeamID'])

    seasons = [season for season in list(range(sim_season-13, sim_season+1)) if season != 2020]
    glm_quality = pd.concat([team_quality(season, regular_season_effects) for season in seasons]).reset_index(drop=True)

    glm_quality_T1 = glm_quality.copy()
    glm_quality_T2 = glm_quality.copy()
    glm_quality_T1.columns = ['T1_TeamID', 'T1_quality', 'Season']
    glm_quality_T2.columns = ['T2_TeamID', 'T2_quality', 'Season']
    tourney_data = pd.merge(tourney_data, glm_quality_T1, on = ['Season', 'T1_TeamID'], how = 'left')
    tourney_data = pd.merge(tourney_data, glm_quality_T2, on = ['Season', 'T2_TeamID'], how = 'left')
    seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

    seeds_T1 = seeds[['Season', 'TeamID', 'seed']].copy()
    seeds_T2 = seeds[['Season', 'TeamID', 'seed']].copy()
    seeds_T1.columns = ['Season', 'T1_TeamID', 'T1_seed']
    seeds_T2.columns = ['Season', 'T2_TeamID', 'T2_seed']

    tourney_data = pd.merge(tourney_data, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, seeds_T2, on=['Season', 'T2_TeamID'], how='left')

    tourney_data["Seed_diff"] = tourney_data["T1_seed"] - tourney_data["T2_seed"]

    features = list(season_statistics_T1.columns[2:999]) + \
               list(season_statistics_T2.columns[2:999]) + \
               list(seeds_T1.columns[2:999]) + \
               list(seeds_T2.columns[2:999]) + \
               list(last14days_stats_T1.columns[2:999]) + \
               list(last14days_stats_T2.columns[2:999]) + \
               ["Seed_diff"] + ["T1_quality", "T2_quality"]


    sub = generate_submission_file(sim_season)
    sub['Season'] = sub['ID'].apply(lambda x: int(x.split('_')[0]))
    sub["T1_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[1]))
    sub["T2_TeamID"] = sub['ID'].apply(lambda x: int(x.split('_')[2]))
    sub = pd.merge(sub, season_statistics_T1, on=['Season', 'T1_TeamID'], how='left')
    sub = pd.merge(sub, season_statistics_T2, on=['Season', 'T2_TeamID'], how='left')

    sub = pd.merge(sub, glm_quality_T1, on=['Season', 'T1_TeamID'], how='left')

    sub = pd.merge(sub, glm_quality_T2, on=['Season', 'T2_TeamID'], how='left')

    sub = pd.merge(sub, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
    sub = pd.merge(sub, seeds_T2, on=['Season', 'T2_TeamID'], how='left')
    sub = pd.merge(sub, last14days_stats_T1, on=['Season', 'T1_TeamID'], how='left')
    sub = pd.merge(sub, last14days_stats_T2, on=['Season', 'T2_TeamID'], how='left')

    sub["Seed_diff"] = sub["T1_seed"] - sub["T2_seed"]
    return tourney_data, sub, features

def cauchyobj(preds, dtrain):
    labels = dtrain.get_label()
    c = 5000
    x =  preds-labels
    grad = x / (x**2/c**2+1)
    hess = -c**2*(x**2-c**2)/(x**2+c**2)**2
    return grad, hess

def modeling(tourney_data, features, repeat_cv = 3):
    param = {}
    # param['objective'] = 'reg:linear'
    param['eval_metric'] = 'mae'
    param['booster'] = 'gbtree'
    param['eta'] = 0.025  # change to ~0.02 for final run
    param['subsample'] = 0.35
    param['colsample_bytree'] = 0.7
    param['num_parallel_tree'] = 10  # recommend 10
    param['min_child_weight'] = 40
    param['gamma'] = 10
    param['max_depth'] = 3
    param['silent'] = 1

    X = tourney_data[features].values
    y = tourney_data['T1_Score'] - tourney_data['T2_Score']

    dtrain = xgb.DMatrix(X, label = y)

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
    print(val_mae)

    oof_preds = []
    for i in range(repeat_cv):
        print(f"Fold repeater {i}")
        preds = y.copy()
        kfold = KFold(n_splits=5, shuffle=True, random_state=i)
        for train_index, val_index in kfold.split(X, y):
            dtrain_i = xgb.DMatrix(X[train_index], label=y[train_index])
            dval_i = xgb.DMatrix(X[val_index], label=y[val_index])
            model = xgb.train(
                params=param,
                dtrain=dtrain_i,
                num_boost_round=iteration_counts[i],
                verbose_eval=50
            )
            preds[val_index] = model.predict(dval_i)
        oof_preds.append(np.clip(preds, -30, 30))

    val_cv = []
    spline_model = []

    for i in range(repeat_cv):
        dat = list(zip(oof_preds[i], np.where(y > 0, 1, 0)))
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

        val_cv.append(pd.DataFrame({"y": np.where(y > 0, 1, 0), "pred": spline_fit, "season": tourney_data.Season}))
        print(f"adjusted logloss of cvsplit {i}: {log_loss(np.where(y > 0, 1, 0), spline_fit)}")

    val_cv = pd.concat(val_cv)
    print(val_cv.groupby('season').apply(lambda x: log_loss(x.y, x.pred)))

    ## Make the actual model after running through validation testing

    sub_models = []
    for i in range(repeat_cv):
        print(f"Fold repeater {i}")
        sub_models.append(
            xgb.train(
                params=param,
                dtrain=dtrain,
                num_boost_round=int(iteration_counts[i] * 1.05),
                verbose_eval=50
            )
        )

    # Added feature importances
    booster = sub_models[0]
    feature_importance = booster.get_score(importance_type='weight')
    mapped_feature_importance = {features[int(key[1:])]: value for key, value in feature_importance.items()}

    # Display the mapped feature importance
    importances = pd.DataFrame([mapped_feature_importance]).T.reset_index().rename(columns={0: 'score', 'index': 'col'})
    print(importances.sort_values('score', ascending=False))

    return {
        'models': sub_models,
        'spline_model': spline_model
    }

def predict(model_obj, sub, features, repeat_cv = 3, upset_overrides=False):
    sub_models = model_obj['models']
    spline_model = model_obj['spline_model']

    Xsub = sub[features].values
    dtest = xgb.DMatrix(Xsub)
    sub_preds = []
    for i in range(repeat_cv):
        sub_preds.append(np.clip(spline_model[i](np.clip(sub_models[i].predict(dtest), -30, 30)), 0.025, 0.975))
    sub["Pred"] = pd.DataFrame(sub_preds).mean(axis=0)

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
import os

def runner(sim_season, repeat_cv=3, upset_overrides=False):
    '''
    Generate predictions for march madness following previously successful algorithm and data processing. Preprocess the data, fit historical data to an XGboost model, predict game results and save them to a csv
    :param sim_season: The season to generate predictions for
    :param repeat_cv: Models to build and average across (boosted, bagging)
    :param upset_overrides: Force score that no upset will happen for the certain list of matchups that an upset has never happened for
    :return:
    '''
    out_columns = ['ID','Pred','Tournament','Season','T1_TeamID','T2_TeamID']
    tourney_data, sub, features = preprocessing(sim_season)
    model_obj = modeling(tourney_data, features, repeat_cv = repeat_cv)
    sub = predict(model_obj, sub, features, repeat_cv = repeat_cv, upset_overrides=upset_overrides)
    sub = sub.loc[sub.Seed_diff.notnull()][out_columns].copy()
    path = f'./output/madness-2023-1st/{sim_season}/'
    os.makedirs(path,exist_ok=True)
    sub.to_csv(f"{path}predictions.csv", index=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='March Madness Prediction Script')
    parser.add_argument('sim_season', type=int, help='The season to generate predictions for')
    parser.add_argument('--repeat_cv', type=int, default=3, help='Models to build and average across (boosted, bagging)')
    parser.add_argument('--upset_overrides', action='store_true', help='Force score that no upset will happen for the certain list of matchups that an upset has never happened for')

    args = parser.parse_args()
    runner(args.sim_season, args.repeat_cv, args.upset_overrides)







