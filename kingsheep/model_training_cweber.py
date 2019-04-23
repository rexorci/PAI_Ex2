import pandas as pd
from sklearn import tree
import ast
import math
import pickle
import os
import glob

# # Load data

# You will get a number of files to train on. Each file contains one game. below is shown how to parse one game, you will have to train on multiple games.

path = "../example_player/training_data"
all_files = glob.glob(os.path.join(path, "1.csv"))
# all_files = glob.glob(os.path.join(path, "*.csv"))

training_data = []

# load the data into a pandas frames
for file in all_files:
    game_data = pd.read_csv(file, index_col=False)
    reason = game_data.iloc[-1][6]

    # if the reason is found, add it to each line to fill out the blanks
    if type(reason) is str:
        for index, row in game_data.iterrows():
            game_data.loc[index, 'reason'] = reason

    # else there was no reason, implying the game reached the number of iterations
    else:
        for index, row in game_data.iterrows():
            game_data.loc[index, 'reason'] = 'max_iterations'

    training_data.append(game_data)

# preview the final 5 lines
training_data[-1].head()

training_data[-1]

# # Feature selection and Instance selection

# ## Sheep

# calculate x and y distance to wolf


X_sheep = []
Y_sheep = []
number_moves = 0

for game in training_data:

    # we want to learn from the winning player, which is the player with the highest score:
    if game.iloc[-1][4] < game.iloc[-1][5]:
        sheep = 's'
        wolf = 'W'

    elif game.iloc[-1][4] > game.iloc[-1][5]:
        sheep = 'S'
        wolf = 'w'
    else:
        continue

    rhubarb = 'r'
    grass = 'g'

    # for each game state in our training data
    for index, row in game.iterrows():

        # we don't want games that ended because of an error or because the sheep commited suicide
        if row['reason'] not in ('sheep1 eaten', 'sheep2 eaten', 'max_iterations'):
            continue

        # we want to only learn from sheep
        if row['move_made'] == 'player1 wolf' or row['move_made'] == 'player2 wolf':
            continue

        number_moves += 1

        # this is the move that we are learning from this game state
        move = row['move_made']

        # create empty feature array for this game state
        game_features = []

        # turn the field from before the move from a string back to a list
        field = ast.literal_eval(row['field_before'])

        # get positions of sheep, wolf and food items
        food = []
        y = 0
        for field_row in field:
            x = 0
            for item in field_row:
                if item == sheep:
                    sheep_position = (x, y)
                elif item == wolf:
                    wolf_position = (x, y)
                elif item == rhubarb or item == grass:
                    food.append((x, y))
                x += 1
            y += 1
        # feature 1: x-distance wolf
        s_feature1 = sheep_position[0] - wolf_position[0]
        # feature 2: y-distance wolf
        s_feature2 = sheep_position[1] - wolf_position[1]

        # determine closest food:
        food_distance = 1000
        food_goal = None
        for food_item in food:
            distance = abs(food_item[0] - sheep_position[0]) + abs(food_item[1] - sheep_position[1])
            if distance < food_distance:
                food_distance = distance
                food_goal = food_item

        # feature 3: x-distance to food
        s_feature3 = 0
        # feature 4: y-distance to food
        s_feature4 = 0

        if food_goal:
            s_feature3 = sheep_position[0] - food_goal[0]
            s_feature4 = sheep_position[1] - food_goal[1]

    game_features.append(s_feature1)
    game_features.append(s_feature2)
    game_features.append(s_feature3)
    game_features.append(s_feature4)

    # add features and move to X_sheep and Y_sheep
    X_sheep.append(game_features)
    Y_sheep.append(move)
#
# # this prints an example of our feature and outcome vector:
# print(X_sheep[0])
# print(Y_sheep[0])
# print(number_moves)
#
# # ## Wolf
#
# # Explain here in text which feature you used for the wolf, and which data you used for training.
#
# # In[69]:
#
#
# # construct your features here
#
#
# # # Train sheep
#
# # In[70]:
#
#
# sheep_tree = tree.DecisionTreeClassifier()
# sheep_tree = sheep_tree.fit(X_sheep, Y_sheep)
#
# # # Train wolf
#
# # In[71]:
#
#
# wolf_tree = tree.DecisionTreeClassifier()
# wolf_tree = wolf_tree.fit(X_wolf, Y_wolf)
#
# # # Save models to files
#
# # Save your models to files here using pickle. Change the [uzhshortname] to your own UZH shortname. This name needs to match the model that you caller in your python player file.
#
# # In[72]:
#
#
# sheep_filename = '[uzhshortname]_sheep_model.sav'
# wolf_filename = '[uzhshortname]_wolf_model.sav'
#
# pickle.dump(sheep_tree, open(sheep_filename, 'wb'))
# pickle.dump(wolf_tree, open(wolf_filename, 'wb'))
