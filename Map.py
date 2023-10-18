import copy
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class MapEnv(object):

    # Initialize env
    def __init__(self, SIZE=(10,10), PROB_OBS=(0,.5), PROB_FLAME=(.3,.5), HP=((.6,1.8),(2,8)),
                 PROB_FIREINT=((.2,.6),(.2,1),.8), BIRTH=(2,3,3), blank_world=False):
        """
        Args:
            SIZE: size of a side of the square grid
            PROB_OBS: range of probabilities that a given block is an obstacle
            PROB_FLAME: flammable probabilities of (free_space, obstacles), 
                        flammable free space == 'grass', flammable obstacles == 'tree',
                        unburnable free space == 'soil', unburnable obstacles == 'stone'.
            HP: ((min grass HP, max grass HP), (max tree HP, max tree HP))
            PROB_FIREINT: ((min_grass_fire_intensity, max_grass_fire_intensity), 
                           (min_tree_fire_intensity, max_tree_fire_intensity), 
                            is_burned_probability)
            BIRTH: (height of support station & birthplace, 
                    width of support station & birthplace,
                    number of agents)
        """

        # Initialize member variables
        self.SIZE               = SIZE
        self.PROB_OBS           = PROB_OBS
        self.PROB_FLAME         = PROB_FLAME
        self.HP                 = HP
        self._HP_INTERVAL       = 0.2
        self.PROB_FIREINT       = PROB_FIREINT
        self._BURN_INTERVAL     = 0.2
        self.BIRTH              = BIRTH
        self.fresh              = True
        self.finished           = False

        # Initialize task world
        self._setWorld()

    def _setWorld(self):
        # Initialize maps
        """
        Tree:  obstacle=1, flammable=1;
        Stone: obstacle=1, flammable=0;
        Grass: obstacle=0, flammable=1;
        Soil:  obstacle=0, flammable=0;
        """
        if self.BIRTH[2] > self.BIRTH[0]*self.BIRTH[1]:
            raise ValueError("The number of agents excesses birthplace dimensions, no valid map returned! ")
        else:
            self._obstacle_map = self._setObstacle() # 0=passable, 1=obstacle
            self._station_row, self._station_col = 0, 0
            self.station_map, self.agent_map = self._setAgent() # 0=free space, 1=agent

            self._flammable_idx = [[], []] # [[flammable free space (grass) idxs], [flammable obstacle (tree) idxs]]
            self._flammable_map = self._setFlammable() 
            self._hp_map = self._setHP()
            self._fire_map = self._setFire() # fire intensity
            self.obstacle_map, self.flammable_map, self.hp_map, self.fire_map = self._mergeMap()
            self.hp_map_init = self.hp_map.copy()


    # Randomize static obstacles    
    def _setObstacle(self):
        prob  = np.random.triangular(self.PROB_OBS[0], .33*self.PROB_OBS[0]+.66*self.PROB_OBS[1], self.PROB_OBS[1])
        obstacle_map = (np.random.rand(int(self.SIZE[0]), int(self.SIZE[1])) < prob).astype(int)
        return obstacle_map


    # Setup agent birthplace (recharging station) & initial position
    def _setAgent(self):
        [row, col] = [np.random.randint(0, self.SIZE[0]-self.BIRTH[0]+1), 
                      np.random.randint(0, self.SIZE[1]-self.BIRTH[1]+1)]
        self._station_row, self._station_col = row, col
        # Birthplace
        station_map = np.zeros_like(self._obstacle_map)
        station = np.ones((self.BIRTH[0],self.BIRTH[1]))
        station_map[row:row+self.BIRTH[0], col:col+self.BIRTH[1]] = station
        # Intial position
        agent_map = np.zeros_like(self._obstacle_map)
        agent = np.zeros((self.BIRTH[0],self.BIRTH[1]))
        agent_num = self.BIRTH[2]
        indices = np.random.choice(self.BIRTH[0] * self.BIRTH[1], agent_num, replace=False)
        agent.ravel()[indices] = 1
        agent_map[row:row+self.BIRTH[0], col:col+self.BIRTH[1]] = agent
        return station_map, agent_map


    # Randomize flammable grids
    def _setFlammable(self):
        flammable_map = np.zeros_like(self._obstacle_map)
        for i in range(self._obstacle_map.shape[0]):
            for j in range(self._obstacle_map.shape[1]):
                # Free grids
                if self._obstacle_map[i, j] == 0:
                    if np.random.rand() < self.PROB_FLAME[0]:
                        flammable_map[i, j] = 1 
                        self._flammable_idx[0].append([i, j])
                # Obstacles
                elif self._obstacle_map[i, j] == 1:
                    if np.random.rand() < self.PROB_FLAME[1]:
                        flammable_map[i, j] = 1
                        self._flammable_idx[1].append([i, j])
        return flammable_map
    

    # Randomize HP of grass & tree based on flammable_idx
    def _setHP(self):
        hp_map = np.zeros_like(self._flammable_map,dtype=float)
        # grass HP
        grass_hp = np.arange(self.HP[0][0], self.HP[0][1] + self._HP_INTERVAL, self._HP_INTERVAL)
        grass_mean, grass_stddev = np.mean(grass_hp), np.std(grass_hp)
        grass_pdf = norm.pdf(grass_hp, loc=grass_mean, scale=grass_stddev)
        grass_prob = grass_pdf / np.sum(grass_pdf)
        for (i,j) in self._flammable_idx[0]:
            hp_map[i,j] = np.around(np.random.choice(grass_hp, p=grass_prob),decimals=1)
        # tree HP
        tree_hp = np.arange(self.HP[1][0], self.HP[1][1] + self._HP_INTERVAL, self._HP_INTERVAL)
        tree_mean, tree_stddev = np.mean(tree_hp), np.std(tree_hp)
        tree_pdf = norm.pdf(tree_hp, loc=tree_mean, scale=tree_stddev)
        tree_prob = tree_pdf / np.sum(tree_pdf)
        for (i,j) in self._flammable_idx[1]:
            hp_map[i,j] = np.around(np.random.choice(tree_hp, p=tree_prob),decimals=1)
        return hp_map


    # Randomize fire intensity of the flammable (grass & tree) based on flammable_idx
    def _setFire(self):
        fire_map = np.zeros_like(self._flammable_map,dtype=float)
        burn_prob = self.PROB_FIREINT[2]
        # grass burning intensity
        grass_burn = np.arange(self.PROB_FIREINT[0][0], self.PROB_FIREINT[0][1] + self._BURN_INTERVAL, self._BURN_INTERVAL)
        grass_mean, grass_stddev = np.mean(grass_burn), np.std(grass_burn)
        grass_pdf = norm.pdf(grass_burn, loc=grass_mean, scale=grass_stddev)
        grass_prob = grass_pdf / np.sum(grass_pdf)
        for (i,j) in self._flammable_idx[0]:
            if np.random.rand() < burn_prob:
                fire_map[i,j] = np.around(np.random.choice(grass_burn, p=grass_prob),decimals=1)
        # tree burning intensity
        tree_burn = np.arange(self.PROB_FIREINT[1][0], self.PROB_FIREINT[1][1] + self._BURN_INTERVAL, self._BURN_INTERVAL)
        tree_mean, tree_stddev = np.mean(tree_burn), np.std(tree_burn)
        tree_pdf = norm.pdf(tree_burn, loc=tree_mean, scale=tree_stddev)
        tree_prob = tree_pdf / np.sum(tree_pdf)
        for (i,j) in self._flammable_idx[1]:
            if np.random.rand() < burn_prob:           
                fire_map[i,j] = np.around(np.random.choice(tree_burn, p=tree_prob),decimals=1)
        return fire_map
    
    #
    def _mergeMap(self):
        station_obs = np.zeros((self.BIRTH[0],self.BIRTH[1]))
        # Merge obstacle map & birthplace
        obstacle_map = self._obstacle_map
        obstacle_map[self._station_row:self._station_row+self.BIRTH[0], 
                     self._station_col:self._station_col+self.BIRTH[1]] = station_obs
        # Merge flammable map & birthplace
        flammable_map = self._flammable_map
        flammable_map[self._station_row:self._station_row+self.BIRTH[0], 
                      self._station_col:self._station_col+self.BIRTH[1]] = station_obs
        # Merge HP map & birthplace
        hp_map = self._hp_map
        hp_map[self._station_row:self._station_row+self.BIRTH[0], 
               self._station_col:self._station_col+self.BIRTH[1]] = station_obs
        # Merge Fire intensity map & birthplace
        fire_map = self._fire_map
        fire_map[self._station_row:self._station_row+self.BIRTH[0], 
                     self._station_col:self._station_col+self.BIRTH[1]] = station_obs
        return obstacle_map, flammable_map, hp_map, fire_map

    # 
    def getObstacle(self):
        return self.obstacle_map

    def getAgent(self):
        return self.agent_map
    
    def getStation(self):
        return self.station_map 
    
    def getFlammable(self):
        return self.flammable_map

    def getHP(self):
        return self.hp_map
    
    def getHPInit(self):
        return self.hp_map_init
    
    def getFireInt(self):
        return self.fire_map
    
    def getAll(self):
        return self.obstacle_map,  self.agent_map, self.station_map, self.flammable_map, self.hp_map, self.fire_map

    def reset(self):
        self._setWorld()

    # Plot map
    def plotObstacle(self):
        cmap = plt.matplotlib.colors.ListedColormap(['white', 'black'])
        plt.figure(figsize=(self.SIZE[0], self.SIZE[1]))
        plt.imshow(self.obstacle_map, cmap=cmap)
        plt.axis()
        plt.show()

    def plotAgent(self):
        cmap = plt.matplotlib.colors.ListedColormap(['white', 'red'])
        plt.figure(figsize=(self.SIZE[0], self.SIZE[1]))
        plt.imshow(self.agent_map, cmap=cmap)
        plt.axis()
        plt.show()

    def plotStation(self):
        cmap = plt.matplotlib.colors.ListedColormap(['white', 'blue'])
        plt.figure(figsize=(self.SIZE[0], self.SIZE[1]))
        plt.imshow(self.station_map, cmap=cmap)
        plt.axis()
        plt.show()

    def plotFlammable(self):
        cmap = plt.matplotlib.colors.ListedColormap(['white', 'green'])
        plt.figure(figsize=(self.SIZE[0], self.SIZE[1]))
        plt.imshow(self.flammable_map, cmap=cmap)
        plt.axis()
        plt.show()

    def plotEnvMap(self):
        cmap = plt.matplotlib.colors.ListedColormap(['white', 'lightgreen', 'black', 'darkgreen'])
        plt.figure(figsize=(self.SIZE[0], self.SIZE[1]))
        env = self.obstacle_map * 2 + self.flammable_map
        plt.imshow(env, cmap=cmap)
        plt.axis()
        plt.show()

    def plotAgentStation(self):
        cmap = plt.matplotlib.colors.ListedColormap(['white', 'blue', 'red'])
        plt.figure(figsize=(self.SIZE[0], self.SIZE[1]))
        env = self.station_map + self.agent_map
        plt.imshow(env, cmap=cmap)
        plt.axis()
        plt.show()

    def plotAll(self):
        cmap = plt.matplotlib.colors.ListedColormap(['white', 'blue', 'red', 'lightgreen', 'black', 
                                                      'yellow', 'yellow', 'darkgreen'])
        plt.figure(figsize=(self.SIZE[0], self.SIZE[1]))
        env = self.station_map + self.agent_map + self.obstacle_map*4 + self.flammable_map*3
        print(env)
        plt.imshow(env, cmap=cmap)
        plt.axis()
        plt.show()
