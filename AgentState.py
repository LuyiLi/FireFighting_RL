# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import MapEnv
NORTH = 1
SOUTH = 2
EAST = 3
WEST = 4


class Reward(object):
    REWARD_COLLISION_OBSTACLES = -1
    REWARD_COLLISION_AGENTS = -1
    REWARD_SUCCESSFUL_MOVEMENT = 0.05
    REWARD_WATER_SPRAY = -0.2
    REWARD_FIRE_FOUGHT_MAGNIFICATION = 1
    REWARD_EPOCH_SUCCESS = 10
    REWARD_EPOCH_UNSUCCESSFUL = -5
    REWARD_DO_NOTHING = -0.05
    REWARD_WATER_DEPLETION = -1


class AgentState(object):
    def __init__(self, map_env: MapEnv, pos_x, pox_y):
        self.pos_x = pos_x
        self.pos_y = pox_y
        self.map = map_env
        self.bias_x = 0
        self.bias_y = 0
        self.water_reserve = 10
        self.observation_size = 11

    def step(self, action):
        """
        One step of the agent
        @param action: A array with direction(0-4), water range(0-5), water direction(0-4), and help beacon(0,1)
        @return: Agent state, Agent reward
        """
        direction = action[0]
        water_range = action[1]
        water_direction = action[2]
        help_beacon = action[3]

        if direction:
            _, reward = self._move(direction)
        else:
            reward = self._spray(water_direction, water_range, self.map.fire_map)
        if help_beacon:
            self._help_beacon()

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
        dx, dy = self._getXYDirection(direction)

        reward = 0
        move_result = self._checkCollision(dx, dy, self.map.obstacle_map, self.map.agent_map)
        if not move_result:
            self.map.agent_map[self.pos_x][self.pos_y] = 0
            self.pos_x += dx
            self.pos_y += dy
            self.map.agent_map[self.pos_x][self.pos_y] = 1
            reward = Reward.REWARD_SUCCESSFUL_MOVEMENT
        elif move_result == -2:
            reward = Reward.REWARD_COLLISION_OBSTACLES
        elif move_result == -3:
            reward = Reward.REWARD_COLLISION_AGENTS
        # TODO: Add water supply algorithm
        return move_result, reward

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

        top_left = (self.pos_x - self.observation_size // 2, self.pos_y - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        obs_shape = (self.observation_size, self.observation_size)

        obs_map = np.zeros(obs_shape)
        agt_map = np.zeros(obs_shape)
        fir_map = np.zeros(obs_shape)
        fla_map = np.zeros(obs_shape)

        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):
                if i >= self.map.SIZE[0] or i < 0 or j >= self.map.SIZE[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    continue
                if obstacle_map[i, j]:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if agents_map[i, j]:
                    # other agent's position
                    agt_map[i - top_left[0], j - top_left[1]] = agents_map[i, j]
                if fire_map[i, j]:
                    # check if there are fire
                    fir_map[i - top_left[0], j - top_left[1]] = fire_map[i, j]
                if flammable_map[i, j]:
                    # if terrain is flammable
                    fir_map[i - top_left[0], j - top_left[1]] = fire_map[i, j]
        state = [obs_map, fir_map, agt_map, fla_map, self.water_reserve]
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
            water_spray = [[1]]
        elif water_range == 2:
            water_spray = [[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]]
        elif water_range == 3:
            water_spray = [[0.00, 0.00, 0.01, 0.00, 0.00],
                           [0.00, 0.01, 0.10, 0.01, 0.00],
                           [0.01, 0.10, 0.52, 0.10, 0.01],
                           [0.00, 0.01, 0.10, 0.01, 0.00],
                           [0.00, 0.00, 0.01, 0.00, 0.00]]
        elif water_range == 4:
            water_spray = [[0.00, 0.01, 0.02, 0.01, 0.00],
                           [0.01, 0.02, 0.09, 0.02, 0.01],
                           [0.02, 0.09, 0.40, 0.09, 0.02],
                           [0.01, 0.02, 0.09, 0.02, 0.01],
                           [0.00, 0.01, 0.02, 0.01, 0.00]]
        elif water_range == 5:
            water_spray = [[0.00, 0.01, 0.03, 0.01, 0.00],
                           [0.01, 0.03, 0.09, 0.03, 0.01],
                           [0.03, 0.09, 0.32, 0.09, 0.03],
                           [0.01, 0.02, 0.09, 0.03, 0.00],
                           [0.00, 0.01, 0.03, 0.01, 0.00]]
        else:
            return Reward.REWARD_DO_NOTHING

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
