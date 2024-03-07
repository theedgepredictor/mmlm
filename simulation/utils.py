import re
import numpy as np
from pathlib import Path
import pandas as pd
import itertools



ROUND_NAMES = {
    0: 'Play-in Games',
    1: 'First Round',
    2: 'Round of 32',
    3: 'Sweet 16',
    4: 'Elite 8',
    5: 'Final 4',
    6: 'Championship'
}


def get_alt_game_id(game_id):
    alt_game_id = game_id.split('_')
    alt_game_id = '_'.join([alt_game_id[0],
                            alt_game_id[2],
                            alt_game_id[1]
                            ])
    return alt_game_id


def get_round(season, t1_id, t2_id, seedyear_dict_rev):
    round_dict = gen_round_dict()

    s_dict_rev = seedyear_dict_rev[season]
    t1_seed = s_dict_rev[t1_id]
    t2_seed = s_dict_rev[t2_id]

    t1_seednum = int(t1_seed[1:3])
    t2_seednum = int(t2_seed[1:3])

    t1_reg = t1_seed[0]
    t2_reg = t2_seed[0]

    area_dict = {'W':'WX', 'X':'WX', 'Y':'YZ', 'Z':'YZ'}

    t1_area = area_dict.get(t1_reg)
    t2_area = area_dict.get(t2_reg)

    if t1_area != t2_area:
        return 6
    elif t1_reg != t2_reg:
        return 5
    else:
        matchup = f'{t2_seednum}v{t1_seednum}'
        return round_dict.get(matchup)

def gen_round_dict():
    round_dict = {}

    r4 = [[1,16,8,9,5,12,4,13,6,11,3,14,7,10,15,2]]
    for seeds in r4:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 4
            round_dict[str(pair[1])+'v'+str(pair[0])] = 4


    r3 = [[1,16,8,9,5,12,4,13],[6,11,3,14,7,10,15,2]]
    for seeds in r3:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 3
            round_dict[str(pair[1])+'v'+str(pair[0])] = 3

    r2 = [[1,16,8,9],[5,12,4,13],[6,11,3,14],[7,10,15,2]]
    for seeds in r2:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 2
            round_dict[str(pair[1])+'v'+str(pair[0])] = 2

    r1 = [[1,16],[8,9],[5,12],[4,13],[6,11],[3,14],[7,10],[15,2]]
    for seeds in r1:
        for pair in itertools.combinations(seeds,2):
            round_dict[str(pair[0])+'v'+str(pair[1])] = 1
            round_dict[str(pair[1])+'v'+str(pair[0])] = 1

    round_dict['11v11'] = 0
    round_dict['12v12'] = 0
    round_dict['13v13'] = 0
    round_dict['14v14'] = 0
    round_dict['16v16'] = 0
    return round_dict
