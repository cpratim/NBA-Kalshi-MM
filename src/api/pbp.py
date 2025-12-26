
from typing import Any
from pbpstats.client import Client
import json
from pbpstats.resources.enhanced_pbp import *
from config import CACHE_DIR, DATA_DIR
from utils.data import get_season_game_ids
import os
from tqdm import tqdm

from time import sleep



class PBP(object):

    def __init__(
        self, 
    ):
        self.settings = settings = {
            "dir": CACHE_DIR,
            "Boxscore": {
                "source": "file", 
                "data_provider": "stats_nba"
            },
            "Possessions": {
                "source": "file", 
                "data_provider": "stats_nba"
            },
        }
        self.record_stats = [
            "points",
            "rebounds",
            "assists",
            "turnovers",
            "steals",
            "blocks",
            "fouls",
            "timeouts",
            'violations',
            "free_throws_made",
            "free_throws_attempted",
            "free_throws_percentage",
            "three_pointers_made",
            "three_pointers_attempted",
            "three_pointers_percentage",
            "two_pointers_made",
            "two_pointers_attempted",
            "two_pointers_percentage",
            "field_goals_made",
            "field_goals_attempted",
            "field_goals_percentage",
            "substitutions",
        ]
        self.client = Client(settings)


    def build_game_dataset(self, game_id: str):
        player_states = {}
        
        dataset = []
        game = self.client.Game(game_id)

        # print(dir(game))
        home_team, away_team = game.boxscore.data['team']
        home_team_id, away_team_id = (
            home_team['team_id'],
            away_team['team_id'],
        )

        game_state = {
            home_team_id: {
                **{stat: 0 for stat in self.record_stats},
            },
            away_team_id: {
                **{stat: 0 for stat in self.record_stats},
            },
        }
        period_state = {
            home_team_id: {
                **{stat: 0 for stat in self.record_stats},
            },
            away_team_id: {
                **{stat: 0 for stat in self.record_stats},
            },
        }

        player_team_map = game.boxscore.player_team_map

        for possession in game.possessions.items:   
            for event in possession.events:
                min, sec = event.clock.split(':')
                left = 60 * int(min) + float(sec)
                time = (720 - left) + int(event.period - 1) * 720

                event_state = {
                    'period': event.period,
                    'minutes': int(min),
                    'seconds': float(sec),
                    'time': time,
                    'left': left,
                    'score': event.score,
                    'score_margin': event.score_margin,
                    home_team_id: event.score[home_team_id],
                    away_team_id: event.score[away_team_id],
                    'seconds_remaining': event.seconds_remaining,
                    'seconds_since_previous_event': event.seconds_since_previous_event,
                }
                current_players = event.current_players
                # for team, players in current_players.items():
                #     if len(players) != 5:
                #         print(len(players))
                
                for team, players in current_players.items():
                    for player in players:
                        if player not in player_states:
                            player_states[player] = {stat: 0 for stat in self.record_stats}
                            player_states[player]['seconds'] = 0
                        player_states[player]['seconds'] += event.seconds_since_previous_event

                match event:
                    case Ejection():
                        self._process_ejection(event, game_state, period_state, player_states, player_team_map)
                    case EndOfPeriod():
                        self._process_end_of_period(event, game_state, period_state, player_states, player_team_map)
                    case FieldGoal():
                        self._process_field_goal(event, game_state, period_state, player_states, player_team_map)
                    case Foul():
                        self._process_foul(event, game_state, period_state, player_states, player_team_map)
                    case FreeThrow():
                        self._process_free_throw(event, game_state, period_state, player_states, player_team_map)
                    case JumpBall():
                        self._process_jump_ball(event, game_state, period_state, player_states, player_team_map)
                    case Rebound():
                        self._process_rebound(event, game_state, period_state, player_states, player_team_map)
                    case Replay():
                        self._process_replay(event, game_state, period_state, player_states, player_team_map)
                    case StartOfPeriod():
                        self._process_start_of_period(event, game_state, period_state, player_states, player_team_map)
                    case Substitution():
                        self._process_substitution(event, game_state, period_state, player_states, player_team_map)
                    case Timeout():
                        self._process_timeout(event, game_state, period_state, player_states, player_team_map)
                    case Turnover():
                        self._process_turnover(event, game_state, period_state, player_states, player_team_map)
                    case Violation():
                        self._process_violation(event, game_state, period_state, player_states, player_team_map)
                
                home_players, away_players = [], []
                for team, players in current_players.items():
                    # team = player_team_map[team]
                    for player in players:
                        sort_stats = (
                            player_states[player]['points'],
                            player_states[player]['rebounds'],
                            player_states[player]['assists'],
                            player_states[player]['seconds'],
                        )
                        if team == 'home':
                            home_players.append((player, sort_stats))
                        else:
                            away_players.append((player, sort_stats))
                    
                home_players.sort(key=lambda x: x[1], reverse=True)
                away_players.sort(key=lambda x: x[1], reverse=True)

                play_state = {
                    'event_state': {**event_state},
                    'game_state': {
                        home_team_id: {k: v for k, v in game_state[home_team_id].items()},
                        away_team_id: {k: v for k, v in game_state[away_team_id].items()},
                    },
                    'period_state': {
                        home_team_id: {k: v for k, v in period_state[home_team_id].items()},
                        away_team_id: {k: v for k, v in period_state[away_team_id].items()},
                    },
                    'player_states': {
                        home_team_id: {player: {k: v for k, v in player_states[player].items()} for player in current_players[home_team_id]},
                        away_team_id: {player: {k: v for k, v in player_states[player].items()} for player in current_players[away_team_id]},
                    },
                }
    
                dataset.append(play_state)
        
        return dataset

    
    def _process_field_goal(self, event: FieldGoal, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        player = event.data['player1_id']
        if player == 0:
            return

        if 'player2_id' in event.data:
            assist_player = event.data['player2_id'] 
        else:
            assist_player = 0
        
        value = event.shot_value
        team = player_team_map[player]
        # team = player_team_map[team]
        if event.is_made and assist_player != 0:
            game_state[team]['assists'] += 1
            period_state[team]['assists'] += 1
            player_states[assist_player]['assists'] += 1

        prefix = 'three_pointers' if value == 3 else 'two_pointers'
        game_state[team]['field_goals_attempted'] += 1
        game_state[team][f'{prefix}_attempted'] += 1
        period_state[team]['field_goals_attempted'] += 1
        period_state[team][f'{prefix}_attempted'] += 1
        player_states[player]['field_goals_attempted'] += 1
        player_states[player]['field_goals_percentage'] = player_states[player]['field_goals_made'] / player_states[player]['field_goals_attempted']
        player_states[player][f'{prefix}_attempted'] += 1
        player_states[player][f'{prefix}_percentage'] = player_states[player][f'{prefix}_made'] / player_states[player][f'{prefix}_attempted']

        if event.is_made:
            game_state[team]['points'] += value
            player_states[player]['points'] += value 
            period_state[team]['points'] += value
            
            player_states[player]['field_goals_made'] += 1
            period_state[team]['field_goals_made'] += 1
            game_state[team]['field_goals_made'] += 1 # added from Pratim's
            
            player_states[player][f'{prefix}_made'] += 1
            period_state[team][f'{prefix}_made'] += 1
            game_state[team][f'{prefix}_made'] += 1 # added from Pratim's

        game_state[team]['field_goals_percentage'] = game_state[team]['field_goals_made'] / game_state[team]['field_goals_attempted']
        game_state[team][f'{prefix}_percentage'] = game_state[team][f'{prefix}_made'] / game_state[team][f'{prefix}_attempted']
        player_states[player]['field_goals_percentage'] = player_states[player]['field_goals_made'] / player_states[player]['field_goals_attempted'] 
        player_states[player][f'{prefix}_percentage'] = player_states[player][f'{prefix}_made'] / player_states[player][f'{prefix}_attempted']
        period_state[team]['field_goals_percentage'] = period_state[team]['field_goals_made'] / period_state[team]['field_goals_attempted']
        period_state[team][f'{prefix}_percentage'] = period_state[team][f'{prefix}_made'] / period_state[team][f'{prefix}_attempted']
        return 

    def _process_rebound(self, event: Rebound, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        player = event.data['player1_id']
        if player == 0:
            return game_state
        team = player_team_map[player]
        game_state[team]['rebounds'] += 1
        period_state[team]['rebounds'] += 1
        player_states[player]['rebounds'] += 1
        return 

    def _process_free_throw(self, event: FreeThrow, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        player = event.data['player1_id']
        if player == 0:
            return game_state
        team = player_team_map[player]
        game_state[team]['free_throws_attempted'] += 1
        period_state[team]['free_throws_attempted'] += 1
        player_states[player]['free_throws_attempted'] += 1
        if event.is_made:
            game_state[team]['points'] += 1
            game_state[team]['free_throws_made'] += 1

            period_state[team]['points'] += 1
            period_state[team]['free_throws_made'] += 1

            player_states[player]['points'] += 1 
            player_states[player]['free_throws_made'] += 1

        game_state[team]['free_throws_percentage'] = game_state[team]['free_throws_made'] / game_state[team]['free_throws_attempted']
        player_states[player]['free_throws_percentage'] = player_states[player]['free_throws_made'] / player_states[player]['free_throws_attempted']
        period_state[team]['free_throws_percentage'] = period_state[team]['free_throws_made'] / period_state[team]['free_throws_attempted']
        return 

    def _process_foul(self, event: Foul, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        player = event.data['player1_id']
        if player == 0 or player not in player_states:
            return game_state

        team = player_team_map[player]
        game_state[team]['fouls'] += 1
        period_state[team]['fouls'] += 1
        player_states[player]['fouls'] += 1
        return game_state

    def _process_jump_ball(self, event: JumpBall, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        return game_state

    def _process_timeout(self, event: Timeout, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        team = event.data['team_id']
        if team == 0:
            return game_state
        game_state[team]['timeouts'] += 1
        period_state[team]['timeouts'] += 1

        return game_state

    def _process_turnover(self, event: Turnover, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        player = event.data['player1_id']
        if player == 0 or player not in player_states:
            return game_state
        team = player_team_map[player]
        game_state[team]['turnovers'] += 1
        period_state[team]['turnovers'] += 1
        player_states[player]['turnovers'] += 1
        return game_state

    def _process_violation(self, event: Violation, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        player = event.data['player1_id']
        if player == 0:
            return game_state
        team = player_team_map[player]
        game_state[team]['violations'] += 1
        period_state[team]['violations'] += 1
        player_states[player]['violations'] += 1
        
        return game_state

    def _process_substitution(self, event: Substitution, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):

        player = event.data['player1_id']
        if player == 0:
            return game_state
        team = player_team_map[player]
        game_state[team]['substitutions'] += 1
        period_state[team]['substitutions'] += 1
        player_states[player]['substitutions'] += 1
        return game_state

    def _process_ejection(self, event: Ejection, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        return game_state

    def _process_start_of_period(self, event: StartOfPeriod, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        return game_state

    def _process_end_of_period(self, event: EndOfPeriod, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        for team in period_state.keys():
            for stat in self.record_stats:
                period_state[team][stat] = 0
        return game_state

    def _process_replay(self, event: Replay, game_state: dict, period_state: dict, player_states: dict, player_team_map: dict):
        return game_state

    def build_season_pbp(self, season: str):

        game_ids = get_season_game_ids(season)
        existing = os.listdir(os.path.join(DATA_DIR, 'pbp'))
        existing = [file.replace('.json', '') for file in existing]
        existing = [game_id for game_id in existing if game_id in game_ids]
        print(f'Total existing pbp: {len(existing)} / {len(game_ids)}')
        for game_id in tqdm(game_ids):
            if game_id in existing:
                continue
            try:
                pbp = self.build_game_dataset(game_id)
                with open(os.path.join(DATA_DIR, 'pbp', f'{game_id}.json'), 'w') as f:
                    json.dump(pbp, f, indent=4)
                # sleep(.5)
            except Exception as e:
                # print(e)
                # print(f"{game_id} - error")
                continue
        return


if __name__ == "__main__":

    pbp = PBP()
    for season in ['2024-25', '2023-24', '2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']:
        print(f'Processing {season}...')
        pbp.build_season_pbp(season)
    # pbp.build_season_pbp('2019-20')
    # pbp.build_game_dataset('0022200012')
    # print(len(os.listdir('/Users/cpratim/Documents/nba-kalshi-mm/data/pbp')))