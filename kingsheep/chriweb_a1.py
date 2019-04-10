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

"""
import math
from collections import defaultdict
from operator import itemgetter

from config import *


def get_class_name():
    return 'ChrisSearchAlgo'



class ChrisSearchAlgo():
    """Example class for a Kingsheep player"""

    def __init__(self):
        self.name = "CSA"
        self.uzh_shortname = "chriweb"

    def get_player_position(self, figure, field):
        x = [x for x in field if figure in x][0]
        return (field.index(x), x.index(figure))

    # defs for sheep
    def move_sheep(self, player_number, field):
        try:
            if player_number == 1:
                figure = CELL_SHEEP_1
            else:
                figure = CELL_SHEEP_2

            if self.wolf_close(player_number, field):
                return self.run_from_wolf(player_number, field, figure)
            elif self.food_present(field):
                gather_move = self.gather_move_sheep(field, figure, player_number)
                if gather_move is None:
                    # Food is present, but not reachable
                    return self.hunt_enemy_sheep(field, player_number)
                else:
                    return gather_move
            else:
                return self.hunt_enemy_sheep(field, player_number)
        except:
            return MOVE_NONE

    def wolf_close(self, player_number, field):
        if player_number == 1:
            sheep_position = self.get_player_position(CELL_SHEEP_1, field)
            wolf_position = self.get_player_position(CELL_WOLF_2, field)
        else:
            sheep_position = self.get_player_position(CELL_SHEEP_2, field)
            wolf_position = self.get_player_position(CELL_WOLF_1, field)

        if self.manhattan_distance(sheep_position, wolf_position) <= 2:
            return True
        return False

    def run_from_wolf(self, player_number, field, figure):
        if player_number == 1:
            sheep_position = self.get_player_position(CELL_SHEEP_1, field)
            wolf_position = self.get_player_position(CELL_WOLF_2, field)
        else:
            sheep_position = self.get_player_position(CELL_SHEEP_2, field)
            wolf_position = self.get_player_position(CELL_WOLF_1, field)

        # go to tile which is accessible and has longest path to wolf
        valid_sheep_moves = self.get_valid_moves(figure, sheep_position, field)

        # do nothing is also valid!
        valid_sheep_moves.append(sheep_position)

        move_heuristics = []
        for move in valid_sheep_moves:
            move_heuristics.append((move, self.manhattan_distance(move, wolf_position)))
        max_heuristic = max(move_heuristics, key=itemgetter(1))
        # if multiple flee options are equally far away, take the one closer to food in this direction
        if self.food_present(field) and len(max_heuristic) > 1:
            best_options = [x for x in move_heuristics if x[1] == max_heuristic[1]]
            best_goal = min(self.get_possible_sheep_goals(player_number, field),
                            key=lambda x: self.weighted_sort(x[2], x[3]))
            target_coord = min(best_options,
                               key=lambda x: self.manhattan_distance(x[0], (best_goal[0], best_goal[1])))
            return self.determine_move_direction(target_coord[0], field, figure)
        else:
            needed_wolf_moves = [(x[0], x[1], self.manhattan_distance(x, wolf_position)) for x in valid_sheep_moves]
            max_steps = max(needed_wolf_moves, key=itemgetter(2))
            best_options_no_food = [x for x in needed_wolf_moves if x[2] == max_steps[2]]
            # go where most degrees of freedom
            target_coord = max(best_options_no_food,
                               key=lambda x: len(self.get_valid_moves(figure, (x[0], x[1]), field)))
            return self.determine_move_direction((target_coord[0], target_coord[1]), field, figure)

    @staticmethod
    def food_present(field):
        food_present = False

        for line in field:
            for item in line:
                if item == CELL_RHUBARB or item == CELL_GRASS:
                    food_present = True
                    break
        return food_present

    @staticmethod
    def weighted_sort(distance, worth):
        return distance * 1.8 / worth

    def gather_move_sheep(self, field, figure, player_number):
        """
        1. get all targets
        2. calculate heuristics for all
        3. calculate real distance with a* on best heuristic
        4. repeat on all where real distance is bigger than heuristic
        """

        possible_goals = sorted(self.get_possible_sheep_goals(player_number, field),
                                key=lambda x: self.weighted_sort(x[2], x[3]))

        # get points of field where 3 fences/borders
        # get all neighbors of this field with 2 fences (recursive)
        # if wolf close,these fields are off limits
        trap_fields = self.get_trap_fields(field)

        if player_number == 1:
            enemy_wolf_pos = self.get_player_position(CELL_WOLF_2, field)
        else:
            enemy_wolf_pos = self.get_player_position(CELL_WOLF_1, field)

        for goal in possible_goals:
            if [el for el in trap_fields if
                el[0] == (goal[0], goal[1]) and self.manhattan_distance(enemy_wolf_pos, goal) <= el[1] + 1]:
                possible_goals.remove(goal)

        i = 0
        possible_path = []
        for goal in possible_goals:
            reverse_path, true_distance = self.a_star_pathfinding((goal[0], goal[1]), figure, field)
            # handle case, if unreachable goal
            if reverse_path:
                possible_path = reverse_path
            else:
                i += 1
                continue

            if len(possible_goals) == i + 1 or self.weighted_sort(true_distance, goal[3]) <= self.weighted_sort(
                    possible_goals[i + 1][2], possible_goals[i + 1][3]):
                break
            i += 1

        if possible_path:
            return self.determine_move_direction(possible_path[-2], field, figure)
        else:
            return None

    def get_trap_fields(self, field):
        dead_ends = self.get_dead_ends(field)
        trap_fields = []
        for item in dead_ends:
            tunnel = []
            tunnel.append(item[0])
            tunnel.extend(self.is_safe_way_out(item[1], item[0], [], field, 0))
            i = 0
            for el in tunnel:
                trap_fields.append((el, len(tunnel) - i))
                i += 1
        return trap_fields

    def is_safe_way_out(self, coord, lag_coord, tunnel_list, field, security_depth):
        security_depth += 1
        way_out_options = self.get_not_fence_or_wall_neighbors(coord, field)
        if security_depth > 50:
            # should never be reached!
            return tunnel_list
        if len(way_out_options) == 2:
            tunnel_list.append(coord)
            next_tunnel_tile = next(obj for obj in way_out_options if obj != lag_coord)
            return self.is_safe_way_out(next_tunnel_tile, coord, tunnel_list, field, security_depth)
        else:
            return tunnel_list

    def get_dead_ends(self, field):
        dead_ends = []
        y_position = 0
        for line in field:
            x_position = 0
            for item in line:
                if item not in [CELL_EMPTY, CELL_GRASS, CELL_RHUBARB]:
                    x_position += 1
                    continue
                good_neighbors = self.get_not_fence_or_wall_neighbors((y_position, x_position), field)

                if len(good_neighbors) == 1:
                    dead_ends.append(((y_position, x_position), good_neighbors[0]))
                x_position += 1
            y_position += 1
        return dead_ends

    def get_not_fence_or_wall_neighbors(self, coord, field):
        good_neighbors = []
        if not self.is_fence_or_border(coord[0] - 1, coord[1], field):
            good_neighbors.append((coord[0] - 1, coord[1]))
        if not self.is_fence_or_border(coord[0] + 1, coord[1], field):
            good_neighbors.append((coord[0] + 1, coord[1]))
        if not self.is_fence_or_border(coord[0], coord[1] - 1, field):
            good_neighbors.append((coord[0], coord[1] - 1))
        if not self.is_fence_or_border(coord[0], coord[1] + 1, field):
            good_neighbors.append((coord[0], coord[1] + 1))
        return good_neighbors

    @staticmethod
    def is_fence_or_border(row, col, field):
        if row > FIELD_HEIGHT - 1:
            return True
        elif row < 0:
            return True
        elif col > FIELD_WIDTH - 1:
            return True
        elif col < 0:
            return True
        elif field[row][col] == CELL_FENCE:
            return True
        else:
            return False

    def get_possible_sheep_goals(self, player_number, field):
        # contains y_pos, x_pos, heuristic
        possible_goals = []

        if player_number == 1:
            sheep_position = self.get_player_position(CELL_SHEEP_1, field)
            enemy_sheep_label = CELL_SHEEP_2
            enemy_wolf_label = CELL_WOLF_2
            friendly_wolf_label = CELL_WOLF_1
        else:
            sheep_position = self.get_player_position(CELL_SHEEP_2, field)
            enemy_sheep_label = CELL_SHEEP_1
            enemy_wolf_label = CELL_WOLF_1
            friendly_wolf_label = CELL_WOLF_2

        weighted_field = [[0 for x in range(len(field[0]))] for y in range(len(field))]
        y_position = 0
        for line in field:
            x_position = 0
            for item in line:
                if item == CELL_GRASS:
                    weighted_field[y_position][x_position] += 1
                elif item == CELL_RHUBARB:
                    weighted_field[y_position][x_position] += 10
                elif item == enemy_sheep_label:
                    weighted_field[y_position][x_position] += 3
                elif item == enemy_wolf_label:
                    weighted_field[y_position][x_position] += -8
                elif item == friendly_wolf_label:
                    weighted_field[y_position][x_position] += 0
                elif item == CELL_FENCE:
                    weighted_field[y_position][x_position] += -0.2

                if item == CELL_RHUBARB or item == CELL_GRASS:
                    possible_goals.append([y_position, x_position,
                                           self.manhattan_distance((y_position, x_position), sheep_position)])
                x_position += 1
            y_position += 1
        radius_field = self.calculate_worth_in_radius(weighted_field)
        for goal in possible_goals:
            goal.append(radius_field[goal[0]][goal[1]])
        return possible_goals

    @staticmethod
    def calculate_worth_in_radius(weighted_field):
        rows = len(weighted_field)
        cols = len(weighted_field[0])

        def up(index, range):
            if index - range * cols >= 0:
                return index - range * cols

        def down(index, range):
            if index + range * cols < rows * cols:
                return index + range * cols

        def left(index, range):
            if index % cols >= range:
                return index - range

        def right(index, range):
            if index % cols < cols - range:
                return index + range

        def combined(index, funct1, funct2, range1, range2):
            if funct1(index, range1) is not None and funct2(index, range2) is not None:
                return funct1(index, range1) + funct2(index, range2) - index

        flat_map = [item for sublist in weighted_field for item in sublist]
        summed_map = [[0 for _ in range(cols)] for _ in range(rows)]

        # set influence for next-door/next-next door neighbor etc. if equal weight, we have 1/2^n because of
        # number of neighbors
        w_n1 = 0.4  # 0.5
        w_n2 = 0.18  # 0.25
        w_n3 = 0.075  # 0.125
        for i in range(rows * cols):
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

            summed_map[math.floor(i / cols)][i % cols] = c

        flat_summed_map = [item for sublist in summed_map for item in sublist]
        min_sum = min(flat_summed_map)
        max_sum = max(flat_summed_map)
        span = max_sum - min_sum
        norm_summed_map = []
        for row in summed_map:
            norm_summed_map.append([(item - min_sum) / span for item in row])
        return norm_summed_map

    def hunt_enemy_sheep(self, field, player_number):
        # go to the cell where the enemy sheep has the most degrees of freedom
        # don't go to fields with distance = 1 to own wolf (enemy sheep will never go there)
        if player_number == 1:
            my_sheep_pos = self.get_player_position(CELL_SHEEP_1, field)
            my_wolf_pos = self.get_player_position(CELL_WOLF_1, field)
            enemy_sheep_pos = self.get_player_position(CELL_SHEEP_2, field)
            my_sheep_figure = CELL_SHEEP_1
            enemy_sheep_figure = CELL_SHEEP_2
        else:
            my_sheep_pos = self.get_player_position(CELL_SHEEP_2, field)
            my_wolf_pos = self.get_player_position(CELL_WOLF_2, field)
            enemy_sheep_pos = self.get_player_position(CELL_SHEEP_1, field)
            my_sheep_figure = CELL_SHEEP_2
            enemy_sheep_figure = CELL_SHEEP_1

        # assumption: sheep flees where most degrees of freedom
        valid_enemy_sheep_moves = self.get_valid_moves(enemy_sheep_figure, enemy_sheep_pos, field)
        moves_dof = [(move, len(self.get_valid_moves(enemy_sheep_figure, move, field))) for move in
                     valid_enemy_sheep_moves]
        max_dof = max(moves_dof, key=itemgetter(1))
        best_options = [x for x in moves_dof if x[1] == max_dof[1]]
        if len(best_options) > 1:
            # more than one option with same number of degrees of freedom. choose the one farthest away from wolf
            needed_wolf_moves = [(x[0], self.manhattan_distance(x[0], my_wolf_pos)) for x in best_options]
            best_target = max(needed_wolf_moves, key=itemgetter(1))
        else:
            best_target = best_options[0]

        # check if it is better to stay if I'm already next to enemy sheep
        if self.manhattan_distance(my_sheep_pos, enemy_sheep_pos) == 1:
            # minus 1, because my sheep will still block 1 for him
            dof_if_move = len(self.get_valid_moves(enemy_sheep_figure, my_sheep_pos, field)) - 1
            if dof_if_move >= max_dof[1]:
                return MOVE_NONE

        reverse_path, f_score = self.a_star_pathfinding(best_target[0], my_sheep_figure, field)
        return self.determine_move_direction(reverse_path[-2], field, my_sheep_figure)

    # defs for wolf
    def move_wolf(self, player_number, field):
        try:
            if player_number == 1:
                sheep_position = self.get_player_position(CELL_SHEEP_2, field)
                return self.determine_wolf_action(sheep_position, field, CELL_WOLF_1)
            else:
                sheep_position = self.get_player_position(CELL_SHEEP_1, field)
                return self.determine_wolf_action(sheep_position, field, CELL_WOLF_2)
        except:
            return MOVE_NONE

    def determine_wolf_action(self, closest_goal, field, figure):
        reverse_path, f_score = self.a_star_pathfinding(closest_goal, figure, field)
        return self.determine_move_direction(reverse_path[-2], field, figure)

    # neutral defs
    def a_star_pathfinding(self, goal, figure, field):
        start = self.get_player_position(figure, field)
        closed_set = set()
        open_set = {start}
        came_from = {}

        g_score = defaultdict(lambda x: 1000)
        g_score[start] = 0

        f_score = defaultdict(lambda x: 1000)
        f_score[start] = self.manhattan_distance(start, goal)

        while open_set:
            _, current_position = min(((f_score[pos], pos) for pos in open_set), key=itemgetter(0))

            if current_position == goal:
                return self.reconstruct_path(came_from, current_position), f_score[goal]

            open_set.remove(current_position)
            closed_set.add(current_position)

            neighbors = self.get_valid_moves(figure, current_position, field)
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current_position] + self.cost_function_astar(figure, field, neighbor)

                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score[neighbor]:
                    continue

                came_from[neighbor] = current_position
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + self.manhattan_distance(neighbor, goal)

        return None, None

    def cost_function_astar(self, figure, field, neighbor):
        field_item = field[neighbor[0]][neighbor[1]]

        if figure == CELL_WOLF_1:
            return self.wolf_cost_funct(CELL_SHEEP_1, CELL_SHEEP_2, CELL_WOLF_1, field_item, field)
        elif figure == CELL_WOLF_2:
            return self.wolf_cost_funct(CELL_SHEEP_2, CELL_SHEEP_1, CELL_WOLF_2, field_item, field)
        elif figure == CELL_SHEEP_1 or figure == CELL_SHEEP_2:
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

    def determine_move_direction(self, coord, field, figure):
        figure_position = self.get_player_position(figure, field)

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

    def get_valid_moves(self, figure, position, field):
        valid_moves = []
        if self.valid_move(figure, position[0] - 1, position[1], field):
            valid_moves.append((position[0] - 1, position[1]))
        if self.valid_move(figure, position[0] + 1, position[1], field):
            valid_moves.append((position[0] + 1, position[1]))
        if self.valid_move(figure, position[0], position[1] + 1, field):
            valid_moves.append((position[0], position[1] + 1))
        if self.valid_move(figure, position[0], position[1] - 1, field):
            valid_moves.append((position[0], position[1] - 1))
        return valid_moves

    @staticmethod
    def reconstruct_path(came_from, current):
        reverse_path = [current]
        while current in came_from:
            current = came_from[current]
            reverse_path.append(current)
        return reverse_path

    @staticmethod
    def manhattan_distance(origin, goal):
        return abs(origin[0] - goal[0]) + abs(origin[1] - goal[1])

    @staticmethod
    def valid_move(figure, x_new, y_new, field):
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
        if figure == CELL_WOLF_1:
            if field[x_new][y_new] == CELL_WOLF_2:
                return False
            elif field[x_new][y_new] == CELL_SHEEP_1:
                return False
        elif figure == CELL_WOLF_2:
            if field[x_new][y_new] == CELL_WOLF_1:
                return False
            elif field[x_new][y_new] == CELL_SHEEP_2:
                return False

        # Sheep can not step on squares occupied by the wolf of the same player.
        # Sheep can not step on squares occupied by the opposite sheep.
        if figure == CELL_SHEEP_1:
            if field[x_new][y_new] == CELL_SHEEP_2 or \
                    field[x_new][y_new] == CELL_WOLF_1 or \
                    field[x_new][y_new] == CELL_WOLF_2:
                return False
        elif figure == CELL_SHEEP_2:
            if field[x_new][y_new] == CELL_SHEEP_1 or \
                    field[x_new][y_new] == CELL_WOLF_2 or \
                    field[x_new][y_new] == CELL_WOLF_1:
                return False

        return True
