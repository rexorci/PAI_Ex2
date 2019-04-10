"""
Kingsheep Agent Template

This template is provided for the course 'Practical Artificial Intelligence' of the University of Zürich. 

Please edit the following things before you upload your agent:
    - change the name of your file to '[uzhshortname]_A2.py', where [uzhshortname] needs to be your uzh shortname
    - change the name of the class to a name of your choosing
    - change the def 'get_class_name()' to return the new name of your class
    - change the init of your class:
        - self.name can be an (anonymous) name of your choosing
        - self.uzh_shortname needs to be your UZH shortname
    - change the name of the model in get_sheep_model to [uzhshortname]_sheep_model
    - change the name of the model in get_wolf_model to [uzhshortname]_wolf_model

The results and rankings of the agents will be published on OLAT using your 'name', not 'uzh_shortname', 
so they are anonymous (and your 'name' is expected to be funny, no pressure).

"""

from config import *
import pickle

def get_class_name():
    return 'MyPlayer'


class MyPlayer():
    """Example class for a Kingsheep player"""

    def __init__(self):
        self.name = "My Player"
        self.uzh_shortname = "mplayer"

    def get_sheep_model(self):
        return pickle.load(open('mplayer_sheep_model.sav','rb'))

    def get_wolf_model(self):
        return pickle.load(open('mplayer_wolf_model.sav','rb'))

    def move_sheep(self, figure, field, sheep_model):
        X_sheep = []
        game_features = []
        
        #preprocess field to get features, add to X_sheep
        #this code is largely copied from the Jupyter Notebook where the models were trained
        
        #create empty feature array for this game state
        
        #add features and move to X_sheep 
        X_sheep.append(game_features)

        result = sheep_model.predict(X_sheep)

        return result


    def move_wolf(self, figure, field, wolf_model):
        X_wolf = []
        game_features = []
        
        #preprocess field to get features, add to X_wolf
        #this code is largely copied from the Jupyter Notebook where the models were trained
        
        #create empty feature array for this game state
        
        #add features and move to X_wolf and Y_wolf
        X_wolf.append(game_features)

        result = wolf_model.predict(X_wolf)

        return result