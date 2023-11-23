# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import astar
import MapEnv
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4
water_range = 1


class Reward(object):
    REWARD_COLLISION_OBSTACLES = -0.5
    REWARD_COLLISION_AGENTS = -0.5
    REWARD_SUCCESSFUL_MOVEMENT = 0.1
    REWARD_WATER_SPRAY = -0.05
    REWARD_FIRE_FOUGHT_MAGNIFICATION = 6
    REWARD_EPOCH_SUCCESS = 1
    REWARD_EPOCH_UNSUCCESSFUL = -1
    REWARD_DO_NOTHING = -0.1
    REWARD_WATER_DEPLETION = -0.3
    REWARD_WATER_REFILL = 0.3


def extract_local_map(global_map, center_i, center_j, local_map_size, fill):
    # 计算局部地图的起始和结束索引
    start_row = max(0, center_i - local_map_size // 2)
    end_row = min(global_map.shape[0], center_i + local_map_size // 2 + 1)
    start_col = max(0, center_j - local_map_size // 2)
    end_col = min(global_map.shape[1], center_j + local_map_size // 2 + 1)

    # 初始化局部地图为零矩阵
    if fill == 0:
        local_map = np.zeros((local_map_size, local_map_size))
    else:
        local_map = np.ones((local_map_size, local_map_size))
    

    # 计算在局部地图中的起始和结束索引
    local_start_row = local_map_size // 2 - (center_i - start_row)
    local_end_row = local_start_row + (end_row - start_row)
    local_start_col = local_map_size // 2 - (center_j - start_col)
    local_end_col = local_start_col + (end_col - start_col)

    # 将全局地图中的数据复制到局部地图中
    local_map[local_start_row:local_end_row, local_start_col:local_end_col] = global_map[start_row:end_row, start_col:end_col]

    return local_map

class AgentState(object):
    def __init__(self, map_env: MapEnv, pos_x, pox_y):
        self.pos_x = pos_x
        self.pos_y = pox_y
        self.map = map_env
        self.bias_x = 0
        self.bias_y = 0
        self.water_reserve = 10
        self.max_water_reserve = 10
        self.observation_size = 5
        

    def step(self, action):
        """
        One step of the agent
        @param action: A array with direction(0-4) and agent move or spray water(0,1) * Move in 0 direction means return to home
        @return: Agent state, Agent reward
        """
        direction = action[0]
        water_direction = action[0]
        # help_beacon = action[3]

        if not action[1]:
            _, reward = self._move(direction)
        else:
            reward = self._spray(water_direction, water_range, self.map.fire_map)
        # if help_beacon:
        #     self._help_beacon()
        # TODO: Add terminal code for robot catching fire

        return self.observe(), reward

    def _getXYDirection(self, direction):
        """
        Get the direction of agent in dx and dy
        @param direction: NORTH, EAST, WEST, SOUTH
        @return: dx, dy
        """
        if direction == NORTH:
            dx, dy = 0, -1
        elif direction == SOUTH:
            dx, dy = 0, 1
        elif direction == WEST:
            dx, dy = -1, 0
        elif direction == EAST:
            dx, dy = 1, 0
        else:
            print("Direction ERROR: " + str(direction))
            raise Exception
        return dx, dy

    def _move(self, direction):
        """
        Move the agent in a given direction
        @param direction: Direction as defined above
        @return: 0 if successful, other if unsuccessful; Reward of the movement
        """
        move_result = 0
        reward = 0
        
        if direction == 0:
            if self.water_reserve == self.max_water_reserve:
                reward += Reward.REWARD_DO_NOTHING
            
            if self.pos_x == self.map.water_pose[0] and self.pos_y == self.map.water_pose[1]:
                reward += Reward.REWARD_COLLISION_OBSTACLES
                move_result = -1 
            else:
                a_s = astar.AStar()
                #TODO: Check this
                dx, dy = a_s.Run(self.map.fire_map + self.map.obstacle_map, [self.pos_x, self.pos_y], self.map.water_pose)
                move_result = self._checkCollision(dx, dy, self.map.obstacle_map, self.map.agent_map)
        else:
            dx, dy = self._getXYDirection(direction)
            move_result = self._checkCollision(dx, dy, self.map.obstacle_map, self.map.agent_map)
            
        if not move_result:
            self.map.agent_map[self.pos_x][self.pos_y] = 0
            self.pos_x += dx
            self.pos_y += dy
            self.map.agent_map[self.pos_x][self.pos_y] = 1
            reward += Reward.REWARD_SUCCESSFUL_MOVEMENT
        elif move_result == -2:
            reward += Reward.REWARD_COLLISION_OBSTACLES
        elif move_result == -3:
            reward += Reward.REWARD_COLLISION_AGENTS
        
        if(self.map.station_map[self.pos_x][self.pos_y]):
            if(self.water_reserve < self.max_water_reserve / 3):
                reward += Reward.REWARD_WATER_REFILL * (self.max_water_reserve - self.water_reserve)
            self.water_reserve = self.max_water_reserve
        
        return move_result, reward

    def _move_to(self, x, y):
        self.map.agent_map[self.pos_x][self.pos_y] = 0
        self.pos_x = x
        self.pos_y = y
        self.map.agent_map[self.pos_x][self.pos_y] = 1


    def _checkCollision(self, dx, dy, obstacle_map, agents_map):
        """
        Check if the agent will collide after moving dx, dy
        @param dx: Movement in x direction
        @param dy: Movement in y direction
        @param obstacle_map: Global obstacle map
        @param agents_map: Global agents map
        @return: 0 if no collision, -1 if out of bound, -2 if collision with obstacles, -3 if collision with agents
        """
        new_map_x = self.pos_x + dx + self.bias_x
        new_map_y = self.pos_y + dy + self.bias_y

        # Test if out of boundary
        if new_map_x >= self.map.SIZE[0] or new_map_x < 0 or new_map_y >= self.map.SIZE[1] or new_map_y < 0:
            return -1
        # Test if collide with obstacles
        if obstacle_map[new_map_x][new_map_y]:
            return -2
        # Test if collide with other agents
        if agents_map[new_map_x][new_map_y]:
            return -3

        return 0

    def observe(self):
        """
        The function for the agent to observe its surroundings
        @param: obstacle_map: Global newest obstacle_map
        @return: The state of the agent as the agent state in the documentation
        """
        state = []
        obstacle_map = self.map.getObstacle()
        fire_map = self.map.getFire()
        agents_map = self.map.getAgent()
        flammable_map = self.map.getFlammable()

        # top_left = (self.pos_x - self.observation_size // 2, self.pos_y - self.observation_size // 2)
        # bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        # obs_shape = (self.observation_size, self.observation_size)

        # obs_map = np.ones(obs_shape)
        # agt_map = np.zeros(obs_shape)
        # fir_map = np.zeros(obs_shape)
        # fla_map = np.zeros(obs_shape)

        # for i in range(top_left[0], top_left[0] + self.observation_size):
        #     for j in range(top_left[1], top_left[1] + self.observation_size):
        #         if i >= self.map.SIZE[0] or i < 0 or j >= self.map.SIZE[1] or j < 0:
        #             # out of bounds, just treat as an obstacle
        #             obs_map[i - top_left[0], j - top_left[1]] = 1
        #             continue
        #         if obstacle_map[i, j]:
        #             # obstacles
        #             obs_map[i - top_left[0], j - top_left[1]] = 1
        #         if agents_map[i, j]:
        #             # other agent's position
        #             agt_map[i - top_left[0], j - top_left[1]] = agents_map[i, j]
        #         if fire_map[i, j]:
        #             # check if there are fire
        #             fir_map[i - top_left[0], j - top_left[1]] = fire_map[i, j]
        #         if flammable_map[i, j]:
        #             # if terrain is flammable
        #             fir_map[i - top_left[0], j - top_left[1]] = fire_map[i, j]
                    
                
        obs_map = extract_local_map(obstacle_map, self.pos_x, self.pos_y, self.observation_size, 1)
        fir_map = extract_local_map(fire_map, self.pos_x, self.pos_y, self.observation_size, 0)
        agt_map = extract_local_map(agents_map, self.pos_x, self.pos_y, self.observation_size, 0)
        fla_map = extract_local_map(flammable_map, self.pos_x, self.pos_y, self.observation_size, 0)
        agt_map[int(self.observation_size / 2)][int(self.observation_size / 2)] = 0
        
        full_map = np.stack([obs_map, fir_map, agt_map, fla_map])
        state = [full_map, self.water_reserve]
        # Perhaps we need to select the following expression...
        # state = [[obs_map, fir_map, agt_map, fla_map], [self.water_reserve]]
        return state

    def _spray(self, water_direction, water_range, fire_map):
        """
        The agent spray water in the corresponding direction
        @param water_direction: Direction to spray water
        @param water_range: Range of the water spray
        @param fire_map: The map containing fire information
        @return: The reward of this action
        """
        if not self.water_reserve:
            return Reward.REWARD_WATER_DEPLETION

        if water_direction == 0:
            return Reward.REWARD_DO_NOTHING

        if water_range == 1:
            water_spray = np.array([[1]])
        elif water_range == 2:
            water_spray = np.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])
        elif water_range == 3:
            water_spray = np.array([[0.00, 0.00, 0.01, 0.00, 0.00],
                           [0.00, 0.01, 0.10, 0.01, 0.00],
                           [0.01, 0.10, 0.52, 0.10, 0.01],
                           [0.00, 0.01, 0.10, 0.01, 0.00],
                           [0.00, 0.00, 0.01, 0.00, 0.00]])
        elif water_range == 4:
            water_spray = np.array([[0.00, 0.01, 0.02, 0.01, 0.00],
                           [0.01, 0.02, 0.09, 0.02, 0.01],
                           [0.02, 0.09, 0.40, 0.09, 0.02],
                           [0.01, 0.02, 0.09, 0.02, 0.01],
                           [0.00, 0.01, 0.02, 0.01, 0.00]])
        elif water_range == 5:
            water_spray = np.array([[0.00, 0.01, 0.03, 0.01, 0.00],
                           [0.01, 0.03, 0.09, 0.03, 0.01],
                           [0.03, 0.09, 0.32, 0.09, 0.03],
                           [0.01, 0.02, 0.09, 0.03, 0.00],
                           [0.00, 0.01, 0.03, 0.01, 0.00]])
        else:
            return Reward.REWARD_DO_NOTHING
        # Add an amplifier
        water_spray *= 2 
        # Specify the spraying center
        dx, dy = self._getXYDirection(water_direction)

        center_x = self.pos_x + dx * water_range + self.bias_x
        center_y = self.pos_y + dy * water_range + self.bias_y

        reward = 0
        l = len(water_spray)
        spray_biased_x, spray_biased_y = int(center_x - l/2 + 0.5), int(center_y - l/2 + 0.5)
        for i in range(l):
            for j in range(l):
                x = spray_biased_x + i
                y = spray_biased_y + j
                if x >= self.map.SIZE[0] or x < 0 or y >= self.map.SIZE[1] or y < 0:
                    continue
                else:
                    reward += (fire_map[x][y] - max(0, fire_map[x][y] - water_spray[i][j]))
                    fire_map[x][y] = max(0, fire_map[x][y] - water_spray[i][j])

        reward *= Reward.REWARD_FIRE_FOUGHT_MAGNIFICATION
        self.water_reserve -= 1
        return reward

    def _help_beacon(self):
        return


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
