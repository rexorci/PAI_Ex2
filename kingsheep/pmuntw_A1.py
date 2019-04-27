"""
Kingsheep Agent Template

This template is provided for the course 'Practical Artificial Intelligence' of the University of Zürich. 

Please edit the following things before you upload your agent:
	- change the name of your file to '[uzhshortname]_A1.py', where [uzhshortname] needs to be your uzh shortname
	- change the name of the class to a name of your choosing
	- change the def 'get_class_name()' to return the new name of your class
	- change the init of your class:
		- self.name can be an (anonymous) name of your choosing
		- self.uzh_shortname needs to be your UZH shortname

The results and rankings of the agents will be published on OLAT using your 'name', not 'uzh_shortname', 
so they are anonymous (and your 'name' is expected to be funny, no pressure).

python kingsheep.py resources/test.map -p1m pmuntw_A1 -p1n AStar -p2m pmuntw_A1 -p2n AStar -g


"""

from config import *
import time
from collections import defaultdict
from math import inf
from operator import itemgetter
import random


def get_class_name():
    return 'AStar'


class AStar():
    """Example class for a Kingsheep player"""

    def __init__(self):
        self.name = "Pjotr"
        self.uzh_shortname = "pmuntw"

    def get_player_position(self, figure, field):
        x = [x for x in field if figure in x][0]
        return (field.index(x), x.index(figure))

    def closest_goal(self, player_number, field):
        possible_goals = []

        if player_number == 1:
            sheep_position = self.get_player_position(CELL_SHEEP_1, field)
            enemy_wolf_position = self.get_player_position(CELL_WOLF_2, field)
        else:
            sheep_position = self.get_player_position(CELL_SHEEP_2, field)
            enemy_wolf_position = self.get_player_position(CELL_WOLF_1, field)

        # make list of possible goals

        y_position = 0
        for line in field:
            x_position = 0
            for item in line:
                if item == CELL_RHUBARB or item == CELL_GRASS:
                    coordinate = (y_position, x_position)
                    #exclude food that is not reachable
                    if self.get_degree_of_freedom_for_food(field, coordinate, player_number) == 1:
                        distance_to_wolf = manhattan_distance(coordinate, enemy_wolf_position)
                        if distance_to_wolf > 3:
                            possible_goals.append(coordinate)
                    if self.get_degree_of_freedom_for_food(field, coordinate, player_number) > 1:
                        possible_goals.append(coordinate)
                x_position += 1
            y_position += 1

        if len(possible_goals) == 0:
            # move so that you get not eaten
            return self.find_random_legal_goal_for_sheep(player_number, field)
        # determine closest item and return
        distance = 1000
        dist_goal_dict = defaultdict(lambda: [])
        for possible_goal in possible_goals:
            if (abs(possible_goal[0] - sheep_position[0]) + abs(possible_goal[1] - sheep_position[1])) < distance:
                distance = abs(possible_goal[0] - sheep_position[0]) + abs(possible_goal[1] - sheep_position[1])
                content = field[possible_goal[0]][possible_goal[1]]
                coordinate_and_content = (possible_goal, content)
                dist_goal_dict[distance].append(coordinate_and_content)

        nearest_food = dist_goal_dict[min(dist_goal_dict)]
        rhubard_food = [item for item in nearest_food if item[1] == 'r']
        if len(rhubard_food) > 0:
            return rhubard_food[0][0]
        else:
            return nearest_food[0][0]

        #return final_goal

    def get_degree_of_freedom_for_food(self, field, current_position, figure):
        neighbours = []
        allowed_fields = [CELL_EMPTY, CELL_GRASS, CELL_RHUBARB]
        allowed_fields.extend([CELL_SHEEP_2, CELL_SHEEP_2_d]) if figure == 2 else allowed_fields.extend(
            [CELL_SHEEP_1, CELL_SHEEP_1_d])

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares

            # Get node position
            node_position = (current_position[0] + new_position[0], current_position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(field) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(field[len(field) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if field[node_position[0]][node_position[1]] not in allowed_fields:
                continue

            neighbours.append(node_position)
        return len(neighbours)


    def find_random_legal_goal_for_sheep(self, player_number, field):
        if player_number == 1:
            sheep_position = self.get_player_position(CELL_SHEEP_1, field)
            enemy_wolf_position = self.get_player_position(CELL_WOLF_2, field)
        else:
            sheep_position = self.get_player_position(CELL_SHEEP_2, field)
            enemy_wolf_position = self.get_player_position(CELL_WOLF_1, field)

        forbidden_fields = self.compute_positions_near_wolf(player_number, field)
        possible_fields = []

        #not itself, must be grass, not near wolf
        for y in range(15):
            for x in range(19):
                new_position = (y, x)
                is_forbidden = new_position in forbidden_fields
                is_itself = (new_position[0] == sheep_position[0] and new_position[1] == sheep_position[1])
                is_grass = field[new_position[0]][new_position[1]] == CELL_EMPTY
                if not is_forbidden and not is_itself and is_grass:
                    possible_fields.append(new_position)
                    
        direction_dict = defaultdict(lambda : [])
        middle_point = (7, 8)
        UP_LEFT = 'UPPER_LEFT'
        UP_RIGHT = 'UPPER_RIGHT'
        DOWN_LEFT = 'DOWN_LEFT'
        DOWN_RIGHT = 'DOWN_RIGHT'
        wolf_area = None
        target_area = None
        
        for possible_field in possible_fields:
            if possible_field[0] <= middle_point[0] and possible_field[1] <= middle_point[1]:
                direction_dict[UP_LEFT].append(possible_field)
            elif possible_field[0] <= middle_point[0] and possible_field[1] > middle_point[1]:
                direction_dict[UP_RIGHT].append(possible_field)
            elif possible_field[0] > middle_point[0] and possible_field[1] > middle_point[1]:
                direction_dict[DOWN_RIGHT].append(possible_field)
            else:
                direction_dict[DOWN_LEFT].append(possible_field)

        if enemy_wolf_position[0] <= middle_point[0] and enemy_wolf_position[1] <= middle_point[1]:
            wolf_area = UP_LEFT
            target_area = UP_RIGHT
        elif enemy_wolf_position[0] <= middle_point[0] and enemy_wolf_position[1] > middle_point[1]:
            wolf_area = UP_RIGHT
            target_area = DOWN_RIGHT
        elif enemy_wolf_position[0] > middle_point[0] and enemy_wolf_position[1] > middle_point[1]:
            wolf_area = DOWN_RIGHT
            target_area = DOWN_LEFT
        else:
            wolf_area = DOWN_LEFT
            target_area = UP_LEFT

        df_dict = defaultdict(lambda: [])
        for possible_field in direction_dict[target_area]:
            df = self.get_degree_of_freedom_for_food(field, possible_field, player_number)
            df_dict[df].append(possible_field)

        max_df = max(df_dict)
        number_of_possible_fields = len(df_dict[max_df])
        random_number = random.randint(0, number_of_possible_fields - 1)
        return df_dict[max_df][random_number]


    def compute_move(self, path):
        if path is None or len(path) < 2:
            print('No path available')
            return MOVE_NONE
        next_position = path[1]
        current_position = path[0]
        difference = (next_position[0] - current_position[0], next_position[1] - current_position[1])
        if difference == (0, 1):
            return MOVE_RIGHT
        elif difference == (0, -1):
            return MOVE_LEFT
        elif difference == (1, 0):
            return MOVE_DOWN
        elif difference == (-1, 0):
            return MOVE_UP
        else:
            print('computeMove was not successful. Difference was:', difference)
            return MOVE_NONE

    def move_sheep(self, figure, field):
        try:
            print('CELL_SHEEP_1')
            start_time = time.time()
            if figure == 1:
                my_sheep = CELL_SHEEP_1
            else:
                my_sheep = CELL_SHEEP_2
            player_position = self.get_player_position(my_sheep, field)
            #print('field', field)
            #print('player_position', player_position)
            # print('figure', figure)
            start = player_position
            end = self.closest_goal(figure, field)
            #print('goal', end)

            path = self.a_star(start, end, field, False, figure)
            #print('path sheep', path)

            end_time = time.time()
            print('computation time:', end_time - start_time, 'seconds')
            return self.compute_move(path)
        except TypeError:
            print('ERROR')
            return MOVE_NONE

    def move_wolf(self, figure, field):
        try:
            print('CELL_WOLF_1')

            start_time = time.time()
            if figure == 1:
                my_wolf = CELL_WOLF_1
                enemy_sheep = CELL_SHEEP_2
            else:
                my_wolf = CELL_WOLF_2
                enemy_sheep = CELL_SHEEP_1
            player_position = self.get_player_position(my_wolf, field)
            #print('player_position', player_position)
            # print('figure', figure)
            start = player_position
            end = self.get_player_position(enemy_sheep, field)
            #print('end', end)
            path = self.a_star(start, end, field, True, figure)
            #print('path', path)

            end_time = time.time()
            print('computation time:', end_time - start_time, 'seconds')
            return self.compute_move(path)
        # edit here incl. the return statement
        except TypeError:
            print('ERROR')
            return MOVE_NONE

    def get_neighbours(self, field, current_position, is_wolf, figure):
        neighbours = []
        allowed_fields_sheep = [CELL_EMPTY, CELL_GRASS, CELL_RHUBARB]
        allowed_fields_wolf = [CELL_EMPTY, CELL_GRASS, CELL_RHUBARB]
        allowed_fields_wolf.extend([CELL_SHEEP_2, CELL_SHEEP_2_d]) if figure == 1 else allowed_fields_wolf.extend(
            [CELL_SHEEP_1, CELL_SHEEP_1_d])
        allowed_fields = allowed_fields_wolf if is_wolf else allowed_fields_sheep

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares
            # for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_position[0] + new_position[0], current_position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(field) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(field[len(field) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            # if maze[node_position[0]][node_position[1]] != 0:
            if field[node_position[0]][node_position[1]] not in allowed_fields:
                continue

            if (not is_wolf) and (self.is_position_near_to_wolf(node_position, figure, field)):
                continue

            neighbours.append(node_position)
        return neighbours

    def compute_positions_near_wolf(self, figure, field):
        if figure == 1:
            enemy_wolf = CELL_WOLF_2
        else:
            enemy_wolf = CELL_WOLF_1

        enemy_wolf_position = self.get_player_position(enemy_wolf, field)
        forbidden_fields = [enemy_wolf_position]
        possible_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for possible_move in possible_moves:
            forbidden_field = (possible_move[0] + enemy_wolf_position[0], possible_move[1] + enemy_wolf_position[1])
            forbidden_fields.append(forbidden_field)
        return forbidden_fields

    def is_position_near_to_wolf(self, position, figure, field):
        return position in self.compute_positions_near_wolf(figure, field)

    def a_star(self, start, goal, field, is_wolf, figure, cost_function=lambda x, y: 1):
        closed_set = set()
        open_set = {start}
        came_from = {}

        # print(start, goal, field, cost_function)
        # print(closed_set, open_set, came_from)

        g_score = defaultdict(lambda x: inf)
        g_score[start] = 0

        f_score = defaultdict(lambda x: inf)
        f_score[start] = manhattan_distance(start, goal)

        while open_set:
            _, current_position = min(((f_score[pos], pos) for pos in open_set), key=itemgetter(0))

            if current_position == goal:
                return reconstruct_path(came_from, current_position)
                # return reconstruct_path(came_from, current_position), f_score[goal]

            open_set.remove(current_position)
            closed_set.add(current_position)

            for neighbor in self.get_neighbours(field, current_position, is_wolf, figure):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current_position] + cost_function(field, neighbor)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current_position
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, goal)


def reconstruct_path(came_from, current):
    total_path = [current]
    while current in came_from.keys():
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]


#def manhattan_distance(current_position, end_position):
#    return ((current_position[0] - end_position[0]) ** 2) + ((current_position[1] - end_position[1]) ** 2)


def manhattan_distance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
