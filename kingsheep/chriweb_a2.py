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
from collections import defaultdict
from operator import itemgetter
import math
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
            sheep_label = CELL_SHEEP_1
            wolf_label = CELL_WOLF_2
        else:
            sheep_label = CELL_SHEEP_2
            wolf_label = CELL_WOLF_1

        X_sheep.append(self.get_features_sheep(figure, field, sheep_label, wolf_label))
        result = sheep_model.predict(X_sheep)

        # MOVE_LEFT = -2
        # MOVE_UP = -1
        # MOVE_NONE = 0
        # MOVE_DOWN = 1
        # MOVE_RIGHT = 2
        return result

    def get_features_sheep(self, figure, field, sheep_label, wolf_label):

        # TODO input the values that jupyter would give and check if we also get an error !

         # 1. get weighted map and all food items
        if figure == 1:
            weighted_field, food_items, sheep_position, wolf_position = self.get_all_field_items(CELL_SHEEP_2,
                                                                                                 CELL_WOLF_2,
                                                                                                 CELL_SHEEP_1,
                                                                                                 CELL_WOLF_1,
                                                                                                 field)
        else:
            weighted_field, food_items, sheep_position, wolf_position = self.get_all_field_items(CELL_SHEEP_1,
                                                                                                 CELL_WOLF_1,
                                                                                                 CELL_SHEEP_2,
                                                                                                 CELL_WOLF_2,
                                                                                                 field)

        # 2. calculate the worth of the food_item, taking it's environment into consideration
        weighted_field_radius = self.calc_worth_in_radius(weighted_field)

        worth_dist_food_items = []
        for food_item in food_items:
            # 3. calculate distance to food_item with a-star
            path = self.a_star_pathfinding(food_item, sheep_position, field, sheep_label)
            if path:
                move_direction = self.determine_move_direction(path[-2], sheep_position)
                worth_dist_food_items.append(
                    (len(path) - 1, weighted_field_radius[food_item[0]][food_item[1]], move_direction))
            # distance, move_direction = self.calc_path_to_goal()

        # 4. sort goals by distance and worth (weighted)
        # 5. pick best for every move direction (if existing)
        # 6. add extra values to result, e.g. 100 if wolf is close
        game_features = []
        for direction in (MOVE_LEFT, MOVE_UP, MOVE_NONE, MOVE_DOWN, MOVE_RIGHT):
            game_features.append(
                self.get_best_score_for_sheep_direction(worth_dist_food_items, direction, sheep_label, wolf_label,
                                                        sheep_position, wolf_position, field))
        return game_features

    def move_wolf(self, figure, field, wolf_model):

        # create empty feature array for this game state
        game_features = []
        X_wolf = []

        if figure == 1:
            sheep_label = CELL_SHEEP_2
            wolf_label = CELL_WOLF_1
        else:
            sheep_label = CELL_SHEEP_1
            wolf_label = CELL_WOLF_2

        # add features and move to X_wolf and Y_wolf
        X_wolf.append(self.get_features_wolf(field, sheep_label, wolf_label))

        result = wolf_model.predict(X_wolf)

        return result

    def get_features_wolf(self, field, sheep_label, wolf_label):
        # get positions of sheep, wolf and food items
        row = 0
        for field_row in field:
            col = 0
            for item in field_row:
                if item == sheep_label:
                    sheep_position = (col, row)
                elif item == wolf_label:
                    wolf_position = (col, row)
                col += 1
            row += 1

        path_to_sheep = self.a_star_pathfinding(sheep_position, wolf_position, field, wolf_label)

        if path_to_sheep:
            move_direction = self.determine_move_direction(path_to_sheep[-2], wolf_position)

        game_features = []
        for direction in (MOVE_LEFT, MOVE_UP, MOVE_NONE, MOVE_DOWN, MOVE_RIGHT):
            game_features.append(
                self.get_best_score_for_wolf_direction(move_direction, direction, wolf_label, wolf_position, field))
        return game_features

    @staticmethod
    def manhattan_distance(origin, goal):
        return abs(origin[0] - goal[0]) + abs(origin[1] - goal[1])

    @staticmethod
    def get_all_field_items(enemy_sheep_label, enemy_wolf_label, friendly_sheep_label, friendly_wolf_label,
                            field):
        weighted_field = [[0 for x in range(FIELD_WIDTH)] for y in range(FIELD_HEIGHT)]
        food_items = []
        row = 0
        for field_row in field:
            col = 0
            for item in field_row:
                if item == CELL_GRASS:
                    weighted_field[row][col] += 1
                    food_items.append((row, col))
                elif item == CELL_RHUBARB:
                    weighted_field[row][col] += 10
                    food_items.append((row, col))
                elif item == enemy_sheep_label:
                    weighted_field[row][col] += 3
                elif item == enemy_wolf_label:
                    weighted_field[row][col] += -8
                    wolf_position = (row, col)
                elif item == friendly_sheep_label:
                    sheep_position = (row, col)
                elif item == friendly_wolf_label:
                    weighted_field[row][col] += 0
                elif item == CELL_FENCE:
                    weighted_field[row][col] += -0.2
                col += 1
            row += 1

        return weighted_field, food_items, sheep_position, wolf_position

    @staticmethod
    def calc_worth_in_radius(weighted_field):
        def up(index, range):
            if index - range * FIELD_WIDTH >= 0:
                return index - range * FIELD_WIDTH

        def down(index, range):
            if index + range * FIELD_WIDTH < FIELD_HEIGHT * FIELD_WIDTH:
                return index + range * FIELD_WIDTH

        def left(index, range):
            if index % FIELD_WIDTH >= range:
                return index - range

        def right(index, range):
            if index % FIELD_WIDTH < FIELD_WIDTH - range:
                return index + range

        def combined(index, funct1, funct2, range1, range2):
            if funct1(index, range1) is not None and funct2(index, range2) is not None:
                return funct1(index, range1) + funct2(index, range2) - index

        flat_map = [item for sublist in weighted_field for item in sublist]
        summed_map = [[0 for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]

        # set influence for next-door/next-next door neighbor etc. if equal weight, we have 1/2^n because of
        # number of neighbors
        w_n1 = 0.4  # 0.5
        w_n2 = 0.18  # 0.25
        w_n3 = 0.075  # 0.125

        for i in range(FIELD_HEIGHT * FIELD_WIDTH):
            neighbors1 = [up(i, 1), down(i, 1), left(i, 1), right(i, 1)]
            neighbors2 = [up(i, 2), down(i, 2), left(i, 2), right(i, 2),
                          combined(i, up, left, 1, 1), combined(i, up, right, 1, 1),
                          combined(i, down, left, 1, 1), combined(i, down, right, 1, 1)]
            neighbors3 = [up(i, 3), down(i, 3), left(i, 3), right(i, 3),
                          combined(i, up, left, 1, 2), combined(i, up, right, 1, 2),
                          combined(i, down, left, 1, 2), combined(i, down, right, 1, 2),
                          combined(i, up, left, 2, 1), combined(i, up, right, 2, 1),
                          combined(i, down, left, 2, 1), combined(i, down, right, 2, 1)
                          ]
            neighbors1_sum = sum([flat_map[j] for j in neighbors1 if j is not None])
            neighbors2_sum = sum([flat_map[j] for j in neighbors2 if j is not None])
            neighbors3_sum = sum([flat_map[j] for j in neighbors3 if j is not None])

            c = flat_map[i] + w_n1 * neighbors1_sum + w_n2 * neighbors2_sum + w_n3 * neighbors3_sum

            summed_map[math.floor(i / FIELD_WIDTH)][i % FIELD_WIDTH] = c

        # min-max normalize values
        flat_summed_map = [item for sublist in summed_map for item in sublist]
        min_sum = min(flat_summed_map)
        max_sum = max(flat_summed_map)
        span = max_sum - min_sum
        norm_summed_map = []
        for row in summed_map:
            norm_summed_map.append([(item - min_sum) / span for item in row])
        return norm_summed_map

    def a_star_pathfinding(self, goal, start_position, field, figure_label):
        closed_set = set()
        open_set = {start_position}
        came_from = {}

        g_score = defaultdict(lambda x: 1000)
        g_score[start_position] = 0

        f_score = defaultdict(lambda x: 1000)
        f_score[start_position] = self.manhattan_distance(start_position, goal)

        while open_set:
            _, current_position = min(((f_score[pos], pos) for pos in open_set), key=itemgetter(0))

            if current_position == goal:
                return self.reconstruct_path(came_from, current_position)

            open_set.remove(current_position)
            closed_set.add(current_position)

            neighbors = self.get_valid_moves(figure_label, current_position, field)
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current_position] + self.cost_function_astar(figure_label, field, neighbor)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current_position
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.manhattan_distance(neighbor, goal)

        return None

    @staticmethod
    def reconstruct_path(came_from, current):
        reverse_path = [current]
        while current in came_from:
            current = came_from[current]
            reverse_path.append(current)
        return reverse_path

    def get_valid_moves(self, figure_label, position, field):
        valid_moves = []
        if self.valid_move(figure_label, position[0] - 1, position[1], field):
            valid_moves.append((position[0] - 1, position[1]))
        if self.valid_move(figure_label, position[0] + 1, position[1], field):
            valid_moves.append((position[0] + 1, position[1]))
        if self.valid_move(figure_label, position[0], position[1] + 1, field):
            valid_moves.append((position[0], position[1] + 1))
        if self.valid_move(figure_label, position[0], position[1] - 1, field):
            valid_moves.append((position[0], position[1] - 1))
        return valid_moves

    @staticmethod
    def valid_move(figure_label, x_new, y_new, field):
        # Neither the sheep nor the wolf, can step on a square outside the map. Imagine the map is surrounded by fences.
        if x_new > FIELD_HEIGHT - 1:
            return False
        elif x_new < 0:
            return False
        elif y_new > FIELD_WIDTH - 1:
            return False
        elif y_new < 0:
            return False

        # Neither the sheep nor the wolf, can enter a square with a fence on.
        if field[x_new][y_new] == CELL_FENCE:
            return False

        # Wolfs can not step on squares occupied by the opponents wolf (wolfs block each other).
        # Wolfs can not step on squares occupied by the sheep of the same player .
        if figure_label == CELL_WOLF_1:
            if field[x_new][y_new] == CELL_WOLF_2:
                return False
            elif field[x_new][y_new] == CELL_SHEEP_1:
                return False
        elif figure_label == CELL_WOLF_2:
            if field[x_new][y_new] == CELL_WOLF_1:
                return False
            elif field[x_new][y_new] == CELL_SHEEP_2:
                return False

        # Sheep can not step on squares occupied by the wolf of the same player.
        # Sheep can not step on squares occupied by the opposite sheep.
        if figure_label == CELL_SHEEP_1:
            if field[x_new][y_new] == CELL_SHEEP_2 or \
                    field[x_new][y_new] == CELL_WOLF_1 or \
                    field[x_new][y_new] == CELL_WOLF_2:
                return False
        elif figure_label == CELL_SHEEP_2:
            if field[x_new][y_new] == CELL_SHEEP_1 or \
                    field[x_new][y_new] == CELL_WOLF_2 or \
                    field[x_new][y_new] == CELL_WOLF_1:
                return False

        return True

    def cost_function_astar(self, figure_label, field, neighbor):
        field_item = field[neighbor[0]][neighbor[1]]

        if figure_label == CELL_WOLF_1:
            return self.wolf_cost_funct(CELL_SHEEP_1, CELL_SHEEP_2, CELL_WOLF_1, field_item, field)
        elif figure_label == CELL_WOLF_2:
            return self.wolf_cost_funct(CELL_SHEEP_2, CELL_SHEEP_1, CELL_WOLF_2, field_item, field)
        elif figure_label == CELL_SHEEP_1 or figure_label == CELL_SHEEP_2:
            if field_item == CELL_GRASS:
                return 0.9
            elif field_item == CELL_RHUBARB:
                return 0.7
            else:
                return 1
        else:
            return 1

    def wolf_cost_funct(self, my_sheep, enemy_sheep, my_wolf, field_item, field):
        # wolf should not step on food if friendly sheep is close
        # wolf should step on food if enemy sheep is close
        my_sheep_pos = self.get_player_position(my_sheep, field)
        enemy_sheep_pos = self.get_player_position(enemy_sheep, field)
        my_pos = self.get_player_position(my_wolf, field)

        dist_to_mine = self.manhattan_distance(my_pos, my_sheep_pos)
        dist_to_enemy = self.manhattan_distance(my_pos, enemy_sheep_pos)

        if dist_to_mine < dist_to_enemy:
            if field_item == CELL_GRASS:
                return 1.1
            elif field_item == CELL_RHUBARB:
                return 1.3
            else:
                return 1
        elif dist_to_mine > dist_to_enemy:
            if field_item == CELL_GRASS:
                return 0.9
            elif field_item == CELL_RHUBARB:
                return 0.7
            else:
                return 1
        else:
            return 1

    @staticmethod
    def get_player_position(figure, field):
        x = [x for x in field if figure in x][0]
        return field.index(x), x.index(figure)

    @staticmethod
    def determine_move_direction(coord, figure_position):
        distance_x = figure_position[1] - coord[1]
        distance_y = figure_position[0] - coord[0]

        if distance_x == 1:
            return MOVE_LEFT
        elif distance_x == -1:
            return MOVE_RIGHT
        elif distance_y == 1:
            return MOVE_UP
        elif distance_y == -1:
            return MOVE_DOWN
        else:
            return MOVE_NONE

    @staticmethod
    def determine_coord_after_move(direction, figure_position):
        if direction == MOVE_LEFT:
            return figure_position[0], figure_position[1] - 1
        elif direction == MOVE_RIGHT:
            return figure_position[0], figure_position[1] + 1
        elif direction == MOVE_UP:
            return figure_position[0] - 1, figure_position[1]
        elif direction == MOVE_DOWN:
            return figure_position[0] + 1, figure_position[1]
        else:
            return figure_position[0], figure_position[1]

    def get_best_score_for_sheep_direction(self, food_items, move_direction, sheep_label, wolf_label, sheep_position,
                                           wolf_position, field):

        coord_after_move = self.determine_coord_after_move(move_direction, sheep_position)
        if self.valid_move(sheep_label, coord_after_move[0], coord_after_move[1], field):
            move_score = 0

            goals_in_direction = list(filter(lambda item: item[2] == move_direction, food_items))
            if goals_in_direction:
                sorted_goals = sorted(goals_in_direction, key=lambda x: self.weighted_sort(x[0], x[1]))
                move_score += self.weighted_sort(sorted_goals[0][0], sorted_goals[0][1])
            else:
                # higher number = worse
                move_score += 50

            path_to_wolf = self.a_star_pathfinding(sheep_position, wolf_position, field, wolf_label)
            return path_to_wolf #TODO remove
            if len(path_to_wolf) < 3:
                # if self.manhattan_distance((coord_after_move[0], coord_after_move[1]), wolf_position) < 3:
                move_score += 100

            return move_score
        else:
            return 1000

    def get_best_score_for_wolf_direction(self, best_direction, move_direction, wolf_label, wolf_position, field):
        coord_after_move = self.determine_coord_after_move(move_direction, wolf_position)
        if self.valid_move(wolf_label, coord_after_move[0], coord_after_move[1], field):
            if best_direction == move_direction:
                move_score = 1
            else:
                # higher number = worse
                move_score = 50
            return move_score
        else:
            return 1000

    @staticmethod
    def weighted_sort(distance, worth):
        return distance ** 1.2 / worth
