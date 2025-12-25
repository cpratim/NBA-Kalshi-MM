import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from tqdm import tqdm

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog, BoxScoreTraditionalV3, BoxScoreAdvancedV3

import json
from pprint import pprint
from config import METADATA_DIR, DATA_DIR



class NBAAPI(object):

    def __init__(self):

        with open(os.path.join(METADATA_DIR, 'seasons.json'), 'r') as f:
            self.seasons = json.load(f)
            
    def _parse_dict(self, _dict: dict):

        res = []
        for _d in _dict['data']:
            sub = {}
            for h, v in zip(_dict['headers'], _d):
                sub[h] = v
            res.append(sub)
        return res


    def get_game_log(self, season: str):
        game_log = LeagueGameLog(season=season)
        df = game_log.get_data_frames()[0]
        output_file = os.path.join(
            DATA_DIR,
            'seasons',
            f'{season}.csv'
        )
        df.to_csv(output_file, index=False)
        return output_file

    def get_all_seasons_game_logs(self):
        for season in self.seasons:
            self.get_game_log(season)

    def get_raw_game_boxscore(self, game_id: str):
        advanced_boxscore = BoxScoreAdvancedV3(game_id=game_id)
        traditional_boxscore = BoxScoreTraditionalV3(game_id=game_id)

        traditional_team_stats   = self._parse_dict(traditional_boxscore.team_stats.get_dict())
        advanced_team_stats      = self._parse_dict(advanced_boxscore.team_stats.get_dict())

        traditional_player_stats = self._parse_dict(traditional_boxscore.player_stats.get_dict())
        advanced_player_stats    = self._parse_dict(advanced_boxscore.player_stats.get_dict())

        boxscore = {
            'traditional_team_stats': traditional_team_stats,
            'advanced_team_stats': advanced_team_stats,
            'traditional_player_stats': traditional_player_stats,
            'advanced_player_stats': advanced_player_stats,
        }

        with open(os.path.join(DATA_DIR, 'boxscores', f'{game_id}.json'), 'w') as f:
            json.dump(boxscore, f, indent=4)


    def get_season_game_ids(self, season: str):
        season_file = os.path.join(DATA_DIR, 'seasons', f'{season}.csv')
        df = pd.read_csv(season_file)
        game_ids = []
        for _id in df['GAME_ID'].values:
            game_ids.append('00' + str(_id))
        game_ids = list(set(game_ids))
        return game_ids


    def get_season_raw_game_boxscores(self, season: str):
        game_ids = self.get_season_game_ids(season)
        existing = os.listdir(os.path.join(DATA_DIR, 'boxscores'))
        existing = [file.replace('.json', '') for file in existing]
        existing = [game_id for game_id in existing if game_id in game_ids]
        print(f'Total existing boxscores: {len(existing)} / {len(game_ids)}')
        for game_id in tqdm(game_ids):
            output_file = os.path.join(DATA_DIR, 'boxscores', f'{game_id}.json')
            if os.path.exists(output_file):
                # print(f'{game_id} - already exists')
                continue
            try:
                self.get_raw_game_boxscore(game_id)
                # print(f'{game_id} - success')
            except Exception as e:
                print(f"{game_id} - error")
                continue
            time.sleep(1 + random.uniform(0, 1))


    def parse_raw_game_boxscore(self, game_id: str):
        with open(os.path.join(DATA_DIR, 'boxscores', f'{game_id}.json'), 'r') as f:
            boxscore = json.load(f)

        traditional_team_stats, advanced_team_stats, traditional_player_stats, advanced_player_stats = (
            boxscore['traditional_team_stats'], 
            boxscore['advanced_team_stats'], 
            boxscore['traditional_player_stats'], 
            boxscore['advanced_player_stats'],
        )

        skip_stats = ['gameId', 'teamCity', 'teamId', 'teamName', 'teamSlug', 'teamTricode']
        aggregated_game_stats = {}
        for stat, val in traditional_team_stats.items():
            if stat in skip_stats:
                continue
            aggregated_game_stats[stat] = val
        for stat, val in advanced_team_stats.items():
            if stat in skip_stats or stat in aggregated_game_stats:
                continue
            aggregated_game_stats[stat] = val
        
        output_file = os.path.join(DATA_DIR, 'cleaned', f'{game_id}.json')
        with open(output_file, 'w') as f:
            json.dump(aggregated_game_stats, f, indent=4)


    def parse_season_raw_game_boxscores(self, season: str):
        game_ids = self.get_season_game_ids(season)
        
        for game_id in tqdm(game_ids):
            if not os.path.exists(os.path.join(DATA_DIR, 'boxscores', f'{game_id}.json')):
                continue
            self.parse_raw_game_boxscore(game_id)

    def process_season(self, season: str):
        season_file = os.path.join(DATA_DIR, 'seasons', f'{season}.csv')

        df = pd.read_csv(season_file)

        season_data = {}
        for game_id, date, win in zip(df['GAME_ID'], df['GAME_DATE'], df['WL']):
            game_id = '00' + str(game_id)
            
            boxscore_file = os.path.join(DATA_DIR, 'boxscores', f'{game_id}.json')
            if not os.path.exists(boxscore_file):
                continue
            with open(boxscore_file, 'r') as f:
                boxscore = json.load(f)
            traditional_team_stats, advanced_team_stats, traditional_player_stats, advanced_player_stats = (
                boxscore['traditional_team_stats'], 
                boxscore['advanced_team_stats'], 
                boxscore['traditional_player_stats'], 
                boxscore['advanced_player_stats'],
            )

            home_team_id = traditional_team_stats[0]['teamId']
            away_team_id = traditional_team_stats[1]['teamId']

            season_data[game_id] = {
                'date': date,
                'season': season,
                'teams': {
                    home_team_id: win,
                    away_team_id: 'L' if win == 'W' else 'W',
                },
                'stats': {},
            }

            for team_stats in traditional_team_stats:
                team_id = str(team_stats['teamId'])
                if team_id not in season_data[game_id]['stats']:
                    season_data[game_id]['stats'][team_id] = {
                        'game': {},
                        'players': {},
                    }

                for stat, val in team_stats.items():
                    if type(val) == str or stat == 'teamId':
                        continue
                    season_data[game_id]['stats'][team_id]['game'][stat] = val

            for team_stats in advanced_team_stats:
                team_id = str(team_stats['teamId'])

                for stat, val in team_stats.items():
                    if type(val) == str or stat == 'teamId' or stat in season_data[game_id]['stats'][team_id]['game']:
                        continue
                    season_data[game_id]['stats'][team_id]['game'][stat] = val

            for player_stats in traditional_player_stats:

                if len(player_stats['minutes']) == 0:
                    continue

                
                team_id = str(player_stats['teamId'])
                player_id = str(player_stats['personId'])

                if player_id not in season_data[game_id]['stats'][team_id]['players']:
                    season_data[game_id]['stats'][team_id]['players'][player_id] = {}

                for stat, val in player_stats.items():
                    if type(val) == str or stat == 'teamId' or stat == 'personId':
                        continue
                    season_data[game_id]['stats'][team_id]['players'][player_id][stat] = val
                
                minutes = player_stats['minutes'].split(':')
                seconds = int(minutes[0]) * 60 + int(minutes[1])
                season_data[game_id]['stats'][team_id]['players'][player_id]['seconds'] = seconds
            
            for player_stats in advanced_player_stats:
                if len(player_stats['minutes']) == 0:
                    continue

                team_id = str(player_stats['teamId'])
                player_id = str(player_stats['personId'])

                if player_id not in season_data[game_id]['stats'][team_id]['players']:
                    season_data[game_id]['stats'][team_id]['players'][player_id] = {}

                for stat, val in player_stats.items():
                    if type(val) == str or stat == 'teamId' or stat == 'personId' or stat in season_data[game_id]['stats'][team_id]['players'][player_id]:
                        continue
                    season_data[game_id]['stats'][team_id]['players'][player_id][stat] = val


        output_file = os.path.join(DATA_DIR, 'processed', f'{season}.json')
        with open(output_file, 'w') as f:
            json.dump(season_data, f, indent=4)


if __name__ == "__main__":
    nba_api = NBAAPI()
    # nba_api.process_season('2016-27')
    nba_api.get_season_raw_game_boxscores('2021-22')
    # nba_api.process_season('2024-25')
