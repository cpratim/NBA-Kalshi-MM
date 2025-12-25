from config import DATA_DIR
import pandas as pd
import os


def get_season_game_ids(season: str):
    season_file = os.path.join(DATA_DIR, 'seasons', f'{season}.csv')
    df = pd.read_csv(season_file)
    game_ids = []
    for _id in df['GAME_ID'].values:
        game_ids.append('00' + str(_id))
    game_ids = list(set(game_ids))
    return game_ids