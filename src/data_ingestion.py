import json 

def get_characters():
    # save_api_response()
    with open('./data/characters.json', 'r') as infile:
        characters = json.load(infile)
        return characters['characters']

def get_scenes():
    with open('./data/episodes.json') as infile:
        scenes = []
        episodes = json.load(infile)['episodes']
        for ep in episodes:
            scenes = scenes + ep['scenes']
        
        return scenes 