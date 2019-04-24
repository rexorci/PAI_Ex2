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
    return 'IntrepidIbex'


class IntrepidIbex():
    """Example class for a Kingsheep player"""

    def __init__(self):
        self.name = "Intrepid Ibex"
        self.uzh_shortname = "chriweb"

    def get_sheep_model(self):
        return pickle.load(open('chriweb_sheep_model.sav', 'rb'))

    def get_wolf_model(self):
        return pickle.load(open('chriweb_wolf_model.sav', 'rb'))

    def move_sheep(self, figure, field, sheep_model):
        X_sheep = []

        # preprocess field to get features, add to X_field
        # this code is largely copied from the Jupyter Notebook where the models were trained

        # create empty feature array for this game state
        game_features = []

        if figure == 1:
            sheep = CELL_SHEEP_1
            wolf = CELL_WOLF_2
        else:
            sheep = CELL_SHEEP_2
            wolf = CELL_WOLF_1

        # get positions of sheep, wolf and food items
        food = []
        y = 0

        if figure == 1:
            items, sheep_position, wolf_position = self.get_all_field_items(CELL_SHEEP_2, CELL_WOLF_2,
                                                                            CELL_SHEEP_1, CELL_WOLF_1, field)
        else:
            items, sheep_position, wolf_position = self.get_all_field_items(CELL_SHEEP_1, CELL_WOLF_1,
                                                                            CELL_SHEEP_2, CELL_WOLF_2, field)

        # for field_row in field:
        #     x = 0
        #     for item in field_row:
        #         if item == sheep:
        #             sheep_position = (x, y)
        #             items.append()
        #         elif item == wolf:
        #             wolf_position = (x, y)
        #         elif item == CELL_RHUBARB or item == CELL_GRASS:
        #             food.append((x, y))
        #         x += 1
        #     y += 1

        # feature 1: x-distance wolf
        sf_x_wolf = sheep_position[0] - wolf_position[0]
        # feature 2: y-distance wolf
        sf_y_wolf = sheep_position[1] - wolf_position[1]

        # determine closest food:
        food_distance = 1000
        food_goal = None
        for food_item in food:
            distance = abs(food_item[0] - sheep_position[0]) + abs(food_item[1] - sheep_position[1])
            if distance < food_distance:
                food_distance = distance
                food_goal = food_item

        # feature 3: x-distance to food
        sf_x_food = 0
        # feature 4: y-distance to food
        sf_y_food = 0
        if food_goal:
            sf_x_food = sheep_position[0] - food_goal[0]
            sf_y_food = sheep_position[1] - food_goal[1]

        # # feature 5: is going up allowed?
        # sf_up = self.valid_move(sheep_position[0] - 1, sheep_position[1], field,
        #                         figure, True)
        # # feature 6: is going down allowed?
        # sf_down = self.valid_move(sheep_position[0] + 1, sheep_position[1], field,
        #                           figure, True)
        # # feature 7: is going right allowed?
        # sf_right = self.valid_move(sheep_position[0], sheep_position[1] + 1, field,
        #                            figure, True)
        # # feature 8: is going left allowed?
        # sf_left = self.valid_move(sheep_position[0], sheep_position[1] - 1, field,
        #                           figure, True)

        # TODO KEEP WORKING FROM HERE (adjust to use all items
        # feature: value of moving left
        sf_v_left = self.value_of_move_sheep2(sheep_position[0] - 1, sheep_position[1], field, items, figure)
        # feature: value of moving up
        sf_v_up = self.value_of_move_sheep2(sheep_position[0], sheep_position[1] - 1, sheep_position, wolf_position,
                                            field,
                                            food_goal,
                                            figure)
        # feature: value of not moving
        sf_v_stay = self.value_of_move_sheep2(sheep_position[0], sheep_position[1], sheep_position, wolf_position,
                                              field,
                                              food_goal,
                                              figure)
        # feature: value of moving down
        sf_v_down = self.value_of_move_sheep2(sheep_position[0], sheep_position[1] + 1, sheep_position, wolf_position,
                                              field,
                                              food_goal,
                                              figure)
        # feature: value of moving right
        sf_v_right = self.value_of_move_sheep2(sheep_position[0] + 1, sheep_position[1], sheep_position, wolf_position,
                                               field,
                                               food_goal,
                                               figure)

        # # feature: value of moving left
        # sf_v_left = self.value_of_move_sheep(sheep_position[0] - 1, sheep_position[1], sheep_position, wolf_position,
        #                                      field,
        #                                      food_goal,
        #                                      figure)
        # # feature: value of moving up
        # sf_v_up = self.value_of_move_sheep(sheep_position[0], sheep_position[1] - 1, sheep_position, wolf_position,
        #                                    field,
        #                                    food_goal,
        #                                    figure)
        # # feature: value of not moving
        # sf_v_stay = self.value_of_move_sheep(sheep_position[0], sheep_position[1], sheep_position, wolf_position, field,
        #                                      food_goal,
        #                                      figure)
        # # feature: value of moving down
        # sf_v_down = self.value_of_move_sheep(sheep_position[0], sheep_position[1] + 1, sheep_position, wolf_position,
        #                                      field,
        #                                      food_goal,
        #                                      figure)
        # # feature: value of moving right
        # sf_v_right = self.value_of_move_sheep(sheep_position[0] + 1, sheep_position[1], sheep_position, wolf_position,
        #                                       field,
        #                                       food_goal,
        #                                       figure)

        # game_features.append(sf_x_wolf)
        # game_features.append(sf_y_wolf)
        # game_features.append(sf_x_food)
        # game_features.append(sf_y_food)
        # game_features.append(sf_up)
        # game_features.append(sf_down)
        # game_features.append(sf_right)
        # game_features.append(sf_left)
        game_features.append(sf_v_left)
        game_features.append(sf_v_up)
        game_features.append(sf_v_stay)
        game_features.append(sf_v_down)
        game_features.append(sf_v_right)

        # add features and move to X_sheep and Y_sheep
        X_sheep.append(game_features)

        result = sheep_model.predict(X_sheep)

        # MOVE_LEFT = -2
        # MOVE_UP = -1
        # MOVE_NONE = 0
        # MOVE_DOWN = 1
        # MOVE_RIGHT = 2
        return result

    def move_wolf(self, figure, field, wolf_model):

        # create empty feature array for this game state
        game_features = []
        X_wolf = []

        if figure == 1:
            sheep = CELL_SHEEP_2
            wolf = CELL_WOLF_1
        else:
            sheep = CELL_SHEEP_1
            wolf = CELL_WOLF_2

        # get positions of sheep, wolf and food items
        y = 0
        for field_row in field:
            x = 0
            for item in field_row:
                if item == sheep:
                    sheep_position = (x, y)
                elif item == wolf:
                    wolf_position = (x, y)
                x += 1
            y += 1

        # feature 1: determine if the sheep is above the wolf
        if wolf_position[1] - sheep_position[1] > 0:
            w_feature1 = 1
        else:
            w_feature1 = 0
        game_features.append(w_feature1)

        # feature 2: determine if the sheep is below the wolf
        if wolf_position[1] - sheep_position[1] < 0:
            w_feature2 = 1
        else:
            w_feature2 = 0
        game_features.append(w_feature2)

        # feature 3: determine if the sheep is left of the wolf
        if wolf_position[0] - sheep_position[0] > 0:
            w_feature3 = 1
        else:
            w_feature3 = 0
        game_features.append(w_feature3)

        # feature 4: determine if the sheep is right of the wolf
        if wolf_position[0] - sheep_position[0] < 0:
            w_feature4 = 1
        else:
            w_feature4 = 0
        game_features.append(w_feature4)

        # add features and move to X_wolf and Y_wolf
        X_wolf.append(game_features)

        result = wolf_model.predict(X_wolf)

        return result

    @staticmethod
    def valid_move(new_col, new_row, field, player, is_sheep=True):
        # Neither the sheep nor the wolf, can step on a square outside the map. Imagine the map is surrounded by fences.
        if new_row > FIELD_HEIGHT - 1:
            return False
        elif new_row < 0:
            return False
        elif new_col > FIELD_WIDTH - 1:
            return False
        elif new_col < 0:
            return False

        # Neither the sheep nor the wolf, can enter a square with a fence on.
        if field[new_row][new_col] == CELL_FENCE:
            return False

        if is_sheep:
            # Sheep can not step on squares occupied by the wolf of the same player.
            # Sheep can not step on squares occupied by the opposite sheep.
            if player == 1:
                if field[new_row][new_col] == CELL_SHEEP_2 or \
                        field[new_row][new_col] == CELL_WOLF_1 or \
                        field[new_row][new_col] == CELL_WOLF_2:
                    return False
            else:
                if field[new_row][new_col] == CELL_SHEEP_1 or \
                        field[new_row][new_col] == CELL_WOLF_2 or \
                        field[new_row][new_col] == CELL_WOLF_1:
                    return False
        else:
            # Wolfs can not step on squares occupied by the opponents wolf (wolfs block each other).
            # Wolfs can not step on squares occupied by the sheep of the same player .
            if player == 1:
                if field[new_row][new_col] == CELL_WOLF_2:
                    return False
                elif field[new_row][new_col] == CELL_SHEEP_1:
                    return False
            else:
                if field[new_row][new_col] == CELL_WOLF_1:
                    return False
                elif field[new_row][new_col] == CELL_SHEEP_2:
                    return False
        return True

    def value_of_move_sheep(self, new_col, new_row, sheep_position, wolf_position, field, food_goal, player):
        # TODO give worth to every item. then multiply 1/distance (manhattan) with worth. we end up with 5 different worths.

        if self.valid_move(new_col, new_row, field, player, True):
            move_value = 0

            if self.manhattan_distance((new_col, new_row), wolf_position) < 3:
                move_value -= 50

            if food_goal:
                # set direction of food
                sf_col_food_before = abs(sheep_position[0] - food_goal[0])
                sf_row_food_before = abs(sheep_position[1] - food_goal[1])
                sf_col_food_after = abs(new_col - food_goal[0])
                sf_row_food_after = abs(new_row - food_goal[1])

                if sf_row_food_after > sf_row_food_before:
                    move_value -= 3
                elif sf_row_food_after < sf_row_food_before:
                    move_value += 2
                if sf_col_food_after > sf_col_food_before:
                    move_value -= 3
                elif sf_col_food_after < sf_col_food_before:
                    move_value += 2

            return move_value
        else:
            # never even attempt to go there
            return -1000

    @staticmethod
    def manhattan_distance(origin, goal):
        return abs(origin[0] - goal[0]) + abs(origin[1] - goal[1])

    def get_all_field_items(self, enemy_sheep_label, enemy_wolf_label, friendly_sheep_label, friendly_wolf_label,
                            field):
        items = []

        row = 0
        for field_row in field:
            col = 0
            for item in field_row:
                if item == CELL_GRASS:
                    items.append(((row, col), 1))
                elif item == CELL_RHUBARB:
                    items.append(((row, col), 10))
                elif item == enemy_sheep_label:
                    items.append(((row, col), 3))
                elif item == enemy_wolf_label:
                    wolf_position = (row, col)
                    items.append(((row, col), -8))
                elif item == friendly_sheep_label:
                    sheep_position = (row, col)
                elif item == friendly_wolf_label:
                    items.append(((row, col), 0))
                elif item == CELL_FENCE:
                    items.append(((row, col), -0.2))
                col += 1
            row += 1

        return items, sheep_position, wolf_position
