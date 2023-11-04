# a_star.py

import sys
import time

import numpy as np


import point
import random_map

class AStar:
    def __init__(self, map, start_pos, dest_pos):
        self.map=map
        self.open_set = []
        self.close_set = []
        self.map_h = map.len(map)
        self.map_w = map.len(map[0])
        self.dest_p = point.Point(dest_pos[0], dest_pos[1])
        self.curr_p = point.Point(start_pos[0], start_pos[1])
        self.curr_p.cost = 0
        

    def BaseCost(self, p):
        x_dis = p.x
        y_dis = p.y
        # Distance to start point
        return x_dis + y_dis + (1.414 - 2) * min(x_dis, y_dis)

    def HeuristicCost(self, p):
        x_dis = self.map_w - 1 - p.x
        y_dis = self.map_h - 1 - p.y
        # Distance to end point
        return x_dis + y_dis + (1.414 - 2) * min(x_dis, y_dis)

    def TotalCost(self, p):
        return self.BaseCost(p) + self.HeuristicCost(p)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0:
            return False
        if x >= self.map_w or y >= self.map_h:
            return False
        return not self.map[x][y]

    def IsInPointList(self, p, point_list):
        for point in point_list:
            if point.x == p.x and point.y == p.y:
                return True
        return False

    def IsInOpenList(self, p):
        return self.IsInPointList(p, self.open_set)

    def IsInCloseList(self, p):
        return self.IsInPointList(p, self.close_set)

    def IsStartPoint(self, p):
        return p.x == 0 and p.y ==0

    def IsEndPoint(self, p):
        return p.x == self.dest_p.x and p.y == self.dest_p.y

    def SaveImage(self, plt):
        millis = int(round(time.time() * 1000))
        filename = './' + str(millis) + '.png'
        plt.savefig(filename)

    def ProcessPoint(self, x, y, parent):
        if not self.IsValidPoint(x, y):
            return # Do nothing for invalid point
        p = point.Point(x, y)
        if self.IsInCloseList(p):
            return # Do nothing for visited point
        print('Process Point [', p.x, ',', p.y, ']', ', cost: ', p.cost)
        if not self.IsInOpenList(p):
            p.parent = parent
            p.cost = self.TotalCost(p)
            self.open_set.append(p)

    def SelectPointInOpenList(self):
        index = 0
        selected_index = -1
        min_cost = sys.maxsize
        for p in self.open_set:
            cost = self.TotalCost(p)
            if cost < min_cost:
                min_cost = cost
                selected_index = index
            index += 1
        return selected_index

    def BuildPath(self, p):
        path = []
        prev = [0, 0]
        while True:
            path.insert(0, p) # Insert first
            
            if self.IsStartPoint(p):
                dx = p.x - prev[0]
                dy = p.y - prev[1]
                break
            else:
                prev = [p.x, p.y]
                p = p.parent
        return dx, dy
        # for p in path:
        #     rec = Rectangle((p.x, p.y), 1, 1, color='g')
        #     ax.add_patch(rec)
        #     plt.draw()
        # self.SaveImage(plt)
        # end_time = time.time()
        # print('===== Algorithm finish in', int(end_time-start_time), ' seconds')

    def Run(self):

        self.open_set.append(self.curr_p)

        while True:
            index = self.SelectPointInOpenList()
            if index < 0:
                # print('No path found, algorithm failed!!!')
                return -2, -2
            p = self.open_set[index]
            # rec = Rectangle((p.x, p.y), 1, 1, color='c')
            # ax.add_patch(rec)
            # self.SaveImage(plt)

            if self.IsEndPoint(p):
                return self.BuildPath(p)

            del self.open_set[index]
            self.close_set.append(p)

            # Process all neighbors
            x = p.x
            y = p.y
            # self.ProcessPoint(x-1, y+1, p)
            self.ProcessPoint(x-1, y, p)
            # self.ProcessPoint(x-1, y-1, p)
            self.ProcessPoint(x, y-1, p)
            # self.ProcessPoint(x+1, y-1, p)
            self.ProcessPoint(x+1, y, p)
            # self.ProcessPoint(x+1, y+1, p)
            self.ProcessPoint(x, y+1, p)




