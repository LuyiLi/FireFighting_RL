
import numpy as np
# import map.py
# Test

# map_l = 15
# map_h = 15

"global parameters"

round_count = 0

int_burning_tensity = 0.2           #Initial burning tensity of grid first catch fire
spread_burning_tensity = 2          #The threshold to spread fire to adjacent grid
max_burning_tensity = 4             #Maximun fire tensity
burning_rate = 0.2                  #Burning intensity increase by each round

# int_hp = 100                        #Initial HP value for each grid
hp_loss = 5                         #HP decrease by each round
fall_prob = 0.5                     #Probability of tree to fall down when HP < 1/3



#Map initialisation
# obstacle_map = np.random.randint(1,size=(15, 15))     
# fire_map = np.random.rand(15, 15)*4
# hp_map = np.zeros((15,15))+int_hp 
# flammable_map = np.random.randint(1,size=(15, 15)) 
    
                        





class MapUpdate(object):
    
    def __init__(self,obstacle_map,fire_map,hp_map,flammable_map):
        
        self.MapEnv = MapEnv()
        self.obstacle_map = MapEnv.getObstacle()     
        self.fire_map = MapEnv.getFireInt()
        self.hp_map = MapEnv.getHP()
        self.int_hp = MapEnv.getHPInit()
        self.flammable_map = MapEnv.getFlammable()
        self.map_l = MapEnv.size(1)
        self.map_h = MapEnv.size(2)


    def map_update(self):
        
        map_l = self.map_l
        map_h = self.map_h
        int_hp = self.int_hp
        
#To calculate the natural influence of burning
        for i in range(map_l):
            for j in range(map_h):

                if self.fire_map[i][j] > 0:
                    
                    self.hp_map[i][j] = max(0, self.hp_map[i][j] - hp_loss)
                    
                    if self.hp_map[i][j] > 0:
                        self.fire_map[i][j] = max(max_burning_tensity, self.fire_map[i][j] + burning_rate) #fire increase by the defined rate 
                    else:
                        self.fire_map[i][j] = 0          #Fire stops when HP fall below 0
                        self.flammable_map[i][j] = 0     #Grid become unflammable                                                 
                
                
#To calculate catching fire by the adjacent grids which fire tensity> spread_burning_tensity
                if self.fire_map[i][j] > spread_burning_tensity:
                    
                    for k in range(max(0,i-1),min(map_l-1,i+1)):
                        if self.flammable_map[k][j] != 0 and self.fire_map[k][j] == 0:
                            self.fire_map[k][j] = int_burning_tensity     

                    for k in range(max(0,j-1),min(map_h-1,j+1)):
                        if self.flammable_map[i][k] != 0 and self.fire_map[i][k] == 0:
                            self.fire_map[i][k] = int_burning_tensity                         
                    
#To calculate the influence of map when burning trees fall down
        for i in range(map_l):
            for j in range(map_h):
                
                if self.flammable_map[i][j] != 1:       #Only trees will fall down, tress are flammable
                    continue
                
                elif self.obstacle_map[i][j] != 1:        #Only trees will fall down, tress are unpassable
                    continue
                
                elif self.hp_map[i][j] >= int_hp/3:          #Only when hp< 1/3 will fall down
                    continue
                
                elif np.random.randint(0, 1, size=1) <= fall_prob:  #Consider fall down probability
                    
                    fall_direction = np.random.randint(1,4,size=1)  #Randomize fall down direction: 1-north, 2-east, 3-south, 4-west
                    
                    if fall_direction == 1 and j-1 >= 0:
                        self.obstacle_map[i][j-1] = 1
                        if self.flammable_map[i][j-1] != 0 and self.fire_map[i][j-1] == 0:
                            self.fire_map[i][j-1] = int_burning_tensity 
                            
                        
                    elif fall_direction == 2 and i+1 <= map_l-1:
                        self.obstacle_map[i+1][j] = 1
                        if self.flammable_map[i+1][j] != 0 and self.fire_map[i+1][j] == 0:
                            self.fire_map[i+1][j] = int_burning_tensity 
                        
                    elif fall_direction == 3 and j+1 <= map_h-1:
                        self.obstacle_map[i][j+1] = 1
                        if self.flammable_map[i][j+1] != 0 and self.fire_map[i][j+1] == 0:
                            self.fire_map[i][j+1] = int_burning_tensity

                                                
                    elif fall_direction == 4 and i-1 >= 0:
                        self.obstacle_map[i-1][j] = 1
                        if self.flammable_map[i-1][j] != 0 and self.fire_map[i-1][j] == 0:
                            self.fire_map[i-1][j] = int_burning_tensity
        
        return self.obstacle_map, self.flammable_map, self.hp_map, self.fire_map
                            
    def getObstacle(self):
        return self.obstacle_map
    
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
                
    def round_update(self, round_count):
        
        round_count += 1
        print('The map has been updated for',str(round_count),'rounds')         




