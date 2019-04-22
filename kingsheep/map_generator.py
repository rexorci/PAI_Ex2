import argparse
import os

from config import *
import random


def generate_random_field(name, entropy, cell_type_counts):
    """
    Generate a random kingsheep map.

    Usage:
    >>> default_counts = {
            CELL_FENCE: 10,
            CELL_RHUBARB: 1,
            CELL_GRASS: 8
        }
    >>> generate_random_field(0.5, default_counts)

    :param entropy: A value in the range(0,1]; how close to each other the elements of the same cell type are placed
    :param cell_type_counts: dictionary of cell_types with their count (divided by 4)
    :return: a 2d field array
    """
    field = [[CELL_EMPTY for _ in range(FIELD_WIDTH)] for _ in range(FIELD_HEIGHT)]

    def next_position(position, length, entropy):
        next_position = position + int(entropy * random.randrange(-position, length - position))
        assert next_position in range(length)
        return next_position

    def place_cells(field, cell_type, count, mirror_type=None):
        mirror_type = mirror_type or cell_type
        previous_x = random.randrange(0, round(FIELD_WIDTH))
        previous_y = random.randrange(0, round(FIELD_HEIGHT / 2))
        for _ in range(count):
            for _ in range(int(10 / entropy)):
                x = next_position(previous_x, round(FIELD_WIDTH), entropy)
                y = next_position(previous_y, round(FIELD_HEIGHT / 2), entropy)
                if field[y][x] == CELL_EMPTY:
                    field[y][x] = cell_type

                    field[-(y + 1)][-(x + 1)] = mirror_type
                    previous_x, previous_y = x, y
                    break

    def place_cells2(field, cell_type, count, point_mirror=False, mirror_type=None):
        mirror_type = mirror_type or cell_type
        previous_x = random.randrange(0, round(FIELD_WIDTH / 2))
        previous_y = random.randrange(0, round(FIELD_HEIGHT / 2))
        for _ in range(count):
            for _ in range(int(10 / entropy)):
                x = next_position(previous_x, round(FIELD_WIDTH / 2), entropy)
                y = next_position(previous_y, round(FIELD_HEIGHT / 2), entropy)
                if field[y][x] == CELL_EMPTY:
                    field[y][x] = cell_type
                    if not point_mirror:
                        field[y][-(x + 1)] = cell_type
                        field[-(y + 1)][x] = cell_type
                    field[-(y + 1)][-(x + 1)] = mirror_type
                    previous_x, previous_y = x, y
                    break

    def store_map(name, map):
        """
        Store map to filesystem for kingsheep game

        Usage:  map = generate_random_field(...)
                store_map('any_name',map)

        :param name: name of the file
        :param map: generated map
        """
        with open(os.path.join(f'{name}.map'), 'w') as file:
            for row in map:
                for cell in row:
                    file.write(cell)
                file.write('\n')

    place_cells(field, CELL_SHEEP_1, 1, mirror_type=CELL_SHEEP_2)
    place_cells(field, CELL_WOLF_1, 1, mirror_type=CELL_WOLF_2)

    for cell_type, count in cell_type_counts.items():
        place_cells(field, cell_type, count)
        # place_cells(field, cell_type, count)

    # place_cells(field, CELL_SHEEP_1, 1, point_mirror=True, mirror_type=CELL_SHEEP_2)
    # place_cells(field, CELL_WOLF_1, 1, point_mirror=True, mirror_type=CELL_WOLF_2)

    store_map(name, field)


parser = argparse.ArgumentParser(description="Create Random Map")
parser.add_argument("-n", "--name", help="name of random map")
parser.add_argument("-e", "--entropy", help="name of random map")
parser.add_argument("-ctc", "--cell_type_counts", help="name of random map")
args = parser.parse_args()

name = "resources/random_maps/random_map"

if args.name:
    name = args.name
if args.entropy:
    entropy = args.entropy
else:
    entropy = random.randint(20, 100) / 100
if args.cell_type_counts:
    cell_type_counts = args.cell_type_counts
else:
    cell_type_counts = {
        CELL_FENCE: random.randint(0, 20),
        CELL_RHUBARB: random.randint(0, 10),
        CELL_GRASS: random.randint(1, 25)
    }

generate_random_field(name, entropy, cell_type_counts)
