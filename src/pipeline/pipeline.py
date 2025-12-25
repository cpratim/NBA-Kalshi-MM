import json
import os

from config import DATA_DIR
from utils.data import get_season_game_ids


class Pipeline(object):


    def __init__(self):

        processed_dir = os.path.join(DATA_DIR, 'processed')
        self.game_data = {}
        for file in os.listdir(processed_dir):
            if not file.endswith('.json'):
                continue
            with open(os.path.join(processed_dir, file), 'r') as f:
                self.game_data.update(json.load(f))

    def process_game(self, game_id: str):

        pbp_file = os.path.join(DATA_DIR, 'pbp', f'{game_id}.json')
        if not os.path.exists(pbp_file):
            return
        with open(pbp_file, 'r') as f:
            pbp = json.load(f)

        if game_id not in self.game_data:
            return
        game_data = self.game_data[game_id]
    
        teams = game_data['teams']
        print(teams)

    def process_season(self, season: str):
        
        game_ids = get_season_game_ids(season)
        

if __name__ == '__main__':

    pipeline = Pipeline()


