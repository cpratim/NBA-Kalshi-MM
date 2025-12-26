import json
import os
import pickle
from config import DATA_DIR
from utils.data import get_season_game_ids

from pprint import pprint
from tqdm import tqdm

import numpy as np



PLAYER_STATS_KEYS = [
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "fieldGoalsPercentage",
    "threePointersMade",
    "threePointersAttempted",
    "threePointersPercentage",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "freeThrowsPercentage",
    "reboundsOffensive",
    "reboundsDefensive",
    "reboundsTotal",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "foulsPersonal",
    "points",
    "plusMinusPoints",
    "seconds",
    "estimatedOffensiveRating",
    "offensiveRating",
    "estimatedDefensiveRating",
    "defensiveRating",
    "estimatedNetRating",
    "netRating",
    "assistPercentage",
    "assistToTurnover",
    "assistRatio",
    "offensiveReboundPercentage",
    "defensiveReboundPercentage",
    "reboundPercentage",
    "turnoverRatio",
    "effectiveFieldGoalPercentage",
    "trueShootingPercentage",
    "usagePercentage",
    "estimatedUsagePercentage",
    "estimatedPace",
    "pace",
    "pacePer40",
    "possessions",
    "PIE",
]

TEAM_STATS_KEYS = [
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "fieldGoalsPercentage",
    "threePointersMade",
    "threePointersAttempted",
    "threePointersPercentage",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "freeThrowsPercentage",
    "reboundsOffensive",
    "reboundsDefensive",
    "reboundsTotal",
    "assists",
    "steals",
    "blocks",
    "turnovers",
    "foulsPersonal",
    "points",
    "plusMinusPoints",
    "estimatedOffensiveRating",
    "offensiveRating",
    "estimatedDefensiveRating",
    "defensiveRating",
    "estimatedNetRating",
    "netRating",
    "assistPercentage",
    "assistToTurnover",
    "assistRatio",
    "offensiveReboundPercentage",
    "defensiveReboundPercentage",
    "reboundPercentage",
    "estimatedTeamTurnoverPercentage",
    "turnoverRatio",
    "effectiveFieldGoalPercentage",
    "estimatedOffensiveRating",
    "trueShootingPercentage",
    "usagePercentage",
    "estimatedUsagePercentage",
    "estimatedPace",
    "pace",
    "pacePer40",
    "possessions",
    "PIE",
]



class Pipeline(object):


    def __init__(self):

        processed_dir = os.path.join(DATA_DIR, 'processed')
        self.game_data = {}
        for file in os.listdir(processed_dir):
            if not file.endswith('.json'):
                continue
            with open(os.path.join(processed_dir, file), 'r') as f:
                self.game_data.update(json.load(f))


    def get_team_stats_until_game(self, game_data: dict, team_id: str):
        date = game_data['date']
        season = game_data['season']

        team_stats = { k: 0 for k in TEAM_STATS_KEYS }
        n_games = 0

        wins = 0
        for game, data in self.game_data.items():
            if data['date'] >= date or data['season'] != season or team_id not in data['teams']:
                continue
                
            if data['teams'][team_id] == 'W':
                wins += 1

            game_team_stats = data['stats'][team_id]['game']
            for stat, val in game_team_stats.items():
                if stat not in TEAM_STATS_KEYS:
                    continue
                team_stats[stat] += val
            n_games += 1

        if n_games == 0:
            team_stats = {k: 0 for k in TEAM_STATS_KEYS}
            wins = 0
        else:
            team_stats = {k: team_stats[k] / n_games for k in TEAM_STATS_KEYS}
            wins = wins / n_games

        team_stats['win_percentage'] = wins
        team_stats['games_played'] = n_games
        return team_stats

    def get_player_stats_until_game(self, game_data: dict, team_id: str, player_id: str, lookback: int = 25):
        date = game_data['date']
        player_stats_lookback = []

        n_games = 0
        for game, data in self.game_data.items():
            if data['date'] >= date:
                continue

            player_stats = None
            for team, team_data in data['stats'].items():
                for player, player_data in team_data['players'].items():
                    if player == player_id:
                        player_stats = player_data
                        break
                if player_stats is not None:
                    break
                
            if player_stats is None:
                continue

            player_stats_lookback.append((data['date'], player_stats))

        player_stats_lookback.sort(key=lambda x: x[0])
        if len(player_stats_lookback) >= lookback:
            player_stats_lookback = player_stats_lookback[-lookback:]

        player_stats = {k: 0 for k in PLAYER_STATS_KEYS}
        n_games = 0
        for date, stats in player_stats_lookback:
            for stat, val in stats.items():
                if stat not in PLAYER_STATS_KEYS:
                    continue
                player_stats[stat] += val
            n_games += 1

        if n_games == 0:
            return {k: 0 for k in PLAYER_STATS_KEYS}

        return {k: player_stats[k] / n_games for k in PLAYER_STATS_KEYS}

    def _flatten_dict(self, _dict: dict):
        flat = []
        for k, v in _dict.items():
            if isinstance(v, dict):
                flat.extend(self._flatten_dict(v))
            else:
                flat.append(v)
        return flat

    def process_pbp(self, home_team: str, away_team: str, game_data: dict, pbp: list, flatten: bool = True):
        home_team_stats = self.get_team_stats_until_game(game_data, home_team)
        away_team_stats = self.get_team_stats_until_game(game_data, away_team)

        home_player_hist_stats = {}
        away_player_hist_stats = {}

        processed_pbp = []
        for play in pbp:
            home_score = play['event_state']['score'][home_team]
            away_score = play['event_state']['score'][away_team]

            home_record = home_team_stats['win_percentage']
            away_record = away_team_stats['win_percentage']
            home_games_played = home_team_stats['games_played']
            away_games_played = away_team_stats['games_played']

            event_state = {
                k: v for k, v in play['event_state'].items() if k not in [home_team, away_team] + ['score', 'score_margin']
            }

            margin = play['game_state'][home_team]['points'] - play['game_state'][away_team]['points']
            event_state['margin'] = margin

            home_team_state = play['game_state'][home_team]
            home_team_state['win_percentage'] = home_record
            home_team_state['games_played'] = home_games_played
            home_team_state['points'] = home_score

            away_team_state = play['game_state'][away_team]
            away_team_state['win_percentage'] = away_record
            away_team_state['games_played'] = away_games_played
            away_team_state['points'] = away_score

            home_state_summary = {}
            for k, v in home_team_state.items():
                if 'made' in k or 'attempted' in k:
                    continue
                home_state_summary[k] = v
            away_state_summary = {}
            for k, v in away_team_state.items():
                if 'made' in k or 'attempted' in k:
                    continue
                away_state_summary[k] = v
            
            game_state = {
                'event': event_state,
                'game': {
                    home_team: home_state_summary,
                    away_team: away_state_summary
                },
            }

            home_players = list(play['player_states'][home_team].keys())
            away_players = list(play['player_states'][away_team].keys())

            home_player_game_stats = play['player_states'][home_team]
            away_player_game_stats = play['player_states'][away_team]

            for player in home_players:
                if player not in home_player_hist_stats:
                    home_player_hist_stats[player] = self.get_player_stats_until_game(game_data, home_team, player)
            for player in away_players:
                if player not in away_player_hist_stats:
                    away_player_hist_stats[player] = self.get_player_stats_until_game(game_data, away_team, player)

            curr_home_player_hist_stats = {k: home_player_hist_stats[k] for k in home_players}
            curr_away_player_hist_stats = {k: away_player_hist_stats[k] for k in away_players}

            team_state = {

                home_team: {
                    'team_hist_stats': home_team_stats,
                    'team_game_stats': home_team_state,
                    'team_period_stats': play['period_state'][home_team],
                    'player_game_stats': home_player_game_stats,
                    'player_hist_stats': curr_home_player_hist_stats

                },
                away_team: {
                    'team_hist_stats': away_team_stats,
                    'team_game_stats': away_team_state,
                    'team_period_stats': play['period_state'][away_team],
                    'player_game_stats': away_player_game_stats,
                    'player_hist_stats': curr_away_player_hist_stats
                }
            }

            if flatten:
                game_state = self._flatten_dict(game_state)
                team_state = self._flatten_dict(team_state)

            print(len(game_state), len(team_state))
            processed_pbp.append(
                {
                    'overview': game_state,
                    'teams': team_state
                }
            )

        return processed_pbp


    def process_game(self, game_id: str, flatten: bool = True):
        pbp_file = os.path.join(DATA_DIR, 'pbp', f'{game_id}.json')
        if not os.path.exists(pbp_file):
            return
        with open(pbp_file, 'r') as f:
            pbp = json.load(f)

        if game_id not in self.game_data:
            return
        game_data = self.game_data[game_id]

        teams = list(game_data['teams'].keys())
        home_team, away_team = teams

        home_result = game_data['teams'][home_team]
        away_result = game_data['teams'][away_team]

        home_processed_pbp = self.process_pbp(home_team, away_team, game_data, pbp, flatten)
        away_processed_pbp = self.process_pbp(away_team, home_team, game_data, pbp, flatten)

        if flatten:
            output_file = os.path.join(DATA_DIR, 'dataset', f'{game_id}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'home': {
                        'result': home_result,
                        'pbp': home_processed_pbp
                    },
                    'away': {
                        'result': away_result,
                        'pbp': away_processed_pbp
                    }
                }, f)
            return

        else:
            output_file = os.path.join(DATA_DIR, 'dataset', f'{game_id}.json')

            with open(output_file, 'w') as f:
                json.dump({
                    'home': {
                        'result': home_result,
                        'pbp': home_processed_pbp
                    },
                    'away': {
                        'result': away_result,
                        'pbp': away_processed_pbp
                    }
                }, f, indent=4)
        return


    def process_season(self, season: str):        
        game_ids = get_season_game_ids(season)
        for game_id in tqdm(game_ids):
            try:
                self.process_game(game_id)
            except Exception as e:
                print(f"Error processing game {game_id}: {e}")
                continue


    def load_dataset(self, batch_size: int = 100, dataset_name: str = 'batch_50'):

        dataset_dir = os.path.join(DATA_DIR, 'dataset')
        X, y = [], []
        for i, file in enumerate(tqdm(os.listdir(dataset_dir))):
            if not file.endswith('.pkl'):
                continue
            with open(os.path.join(dataset_dir, file), 'rb') as f:
                data = pickle.load(f)
            
            home_X = np.array(data['home']['pbp'])
            away_X = np.array(data['away']['pbp'])

            home_y = 1 if data['home']['result'] == 'W' else 0
            away_y = 1 if data['away']['result'] == 'W' else 0

            X.append(home_X)
            X.append(away_X)
            y.append(home_y)
            y.append(away_y)

            if (i + 1) % batch_size == 0:
                output_file = os.path.join(DATA_DIR, 'XY', dataset_name, f'batch_{i // batch_size + 1}.pkl')
                with open(output_file, 'wb') as f:
                    pickle.dump({'X': X, 'y': y}, f)
                X, y = [], []

        if len(X) > 0:
            output_file = os.path.join(DATA_DIR, 'XY', dataset_name, f'batch_{i // batch_size + 1}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump({'X': X, 'y': y}, f)
        return


    def load_autoencoder_dataset(self, dataset_name: str = 'batch_50', n_features: int = 830):
        data_dir = os.path.join(DATA_DIR, 'XY', dataset_name)
        for file in tqdm(os.listdir(data_dir)):
            batch_idx = int(file.split('_')[1].split('.')[0])

            output_file = os.path.join(DATA_DIR, 'XY', 'autoencoder', f'batch_{batch_idx}.pkl')
            if os.path.exists(output_file):
                continue
            if not file.endswith('.pkl'):
                continue
            X = []
            with open(os.path.join(data_dir, file), 'rb') as f:
                data = pickle.load(f)
            for x in data['X']:
                for p in x:
                    t = p['teams']
                    if len(t) != n_features:
                        continue
                   
                    X.append(t)
            
            X = np.array(X)
            with open(output_file, 'wb') as f:
                pickle.dump(X, f)


if __name__ == '__main__':

    pipeline = Pipeline()
    # dataset = pipeline.load_dataset()
    # pipeline.process_season('2019-20')
    # dataset = pipeline.load_autoencoder_dataset()
    # dataset = pipeline.load_dataset(batch_size=50, dataset_name='batch_50')
    # pipeline.load_autoencoder_dataset(dataset_name='batch_50')
    pipeline.load_autoencoder_dataset(dataset_name='batch_50')
