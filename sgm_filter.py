
import sys
import time as t
import cv2
import numpy as np


class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (x, y) for cardinal direction.
        :param name: common name of said direction.
        """
        self.direction = direction
        self.name = name

# 8 defined directions for sgm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')

class Paths:
    def __init__(self):
        """
        represent the relation between the directions.
        """
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(self,  P1=5, P2=70):
        """
        represent all parameters used in the sgm algorithm.
        :param max_disparity: maximum distance between the same pixel in both images.
        :param P1: penalty for disparity difference = 1
        :param P2: penalty for disparity difference > 1
        :param csize: size of the kernel for the census transform.
        :param bsize: size of the kernel for blurring the images and median filtering.
        """

        self.P1 = P1
        self.P2 = P2

       
def get_path_cost(slice, offset, parameters):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param parameters: structure containing parameters of the algorithm.
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = parameters.P1
    penalties[np.abs(disparities - disparities.T) > 1] = parameters.P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume, parameters, paths):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]
    start = -(height - 1)
    end = width - 1

    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)

    path_id = 0
    for path in paths.effective_paths:
        print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')
        sys.stdout.flush()
        dawn = t.time()

        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=cost_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, parameters)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, parameters), axis=0)

        if main.direction == E.direction:
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, parameters)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, parameters), axis=0)

        # if main.direction == SE.direction:
        #     for offset in range(start, end):
        #         south_east = cost_volume.diagonal(offset=offset).T
        #         north_west = np.flip(south_east, axis=0)
        #         dim = south_east.shape[0]
        #         y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
        #         y_nw_idx = np.flip(y_se_idx, axis=0)
        #         x_nw_idx = np.flip(x_se_idx, axis=0)
        #         main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, parameters)
        #         opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, parameters)

        # if main.direction == SW.direction:
        #     for offset in range(start, end):
        #         south_west = np.flipud(cost_volume).diagonal(offset=offset).T
        #         north_east = np.flip(south_west, axis=0)
        #         dim = south_west.shape[0]
        #         y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
        #         y_ne_idx = np.flip(y_sw_idx, axis=0)
        #         x_ne_idx = np.flip(x_sw_idx, axis=0)
        #         main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, parameters)
        #         opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, parameters)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return aggregation_volume
