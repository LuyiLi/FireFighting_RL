
import numpy as np


map_l = 15
map_h = 15
round_count = 0

int_burning_tensity = 0.2           #Initial burning tensity of grid first catch fire
spread_burning_tensity = 2          #The threshold to spread fire to adjacent grid
max_burning_tensity = 4             #Maximun fire tensity
burning_rate = 0.2                  #Burning intensity increase by each round

int_hp = 100                        #Initial HP value for each grid
hp_loss = 5                         #HP decrease by each round
fall_prob = 0.5                     #Probability of tree to fall down when HP < 1/3



#Map initialisation
obstacle_map = np.random.randint(1,size=(15, 15))     
fire_map = np.random.rand(15, 15)*4
hp_map = np.zeros((15,15))+int_hp 
flammable_map = np.random.randint(1,size=(15, 15)) 

class MapUpdate(object):


    def map_update(self):
        
#To calculate the natural influence of burning
        for i in range(map_l):
            for j in range(map_h):

                if fire_map[i][j] > 0:
                    
                    hp_map[i][j] = max(0, hp_map[i][j] - hp_loss)
                    
                    if hp_map[i][j] > 0:
                        fire_map[i][j] = max(max_burning_tensity, fire_map[i][j] + burning_rate) #fire increase by the defined rate 
                    else:
                        fire_map[i][j] = 0          #Fire stops when HP fall below 0
                        flammable_map[i][j] = 0     #Grid become unflammable                                                 
                
                
#To calculate catching fire by the adjacent grids which fire tensity> spread_burning_tensity
                if fire_map[i][j] > spread_burning_tensity:
                    
                    for k in range(max(0,i-1),min(map_l-1,i+1)):
                        if flammable_map[k][j] != 0 and fire_map[k][j] == 0:
                            fire_map[k][j] = int_burning_tensity     

                    for k in range(max(0,j-1),min(map_h-1,j+1)):
                        if flammable_map[i][k] != 0 and fire_map[i][k] == 0:
                            fire_map[i][k] = int_burning_tensity                         
                    
#To calculate the influence of map when burning trees fall down
        for i in range(map_l):
            for j in range(map_h):
                
                if flammable_map[i][j] != 1:       #Only trees will fall down, tress are flammable
                    continue
                
                elif obstacle_map[i][j] != 1:        #Only trees will fall down, tress are unpassable
                    continue
                
                elif hp_map[i][j] >= int_hp/3:          #Only when hp< 1/3 will fall down
                    continue
                
                elif np.random.randint(0, 1, size=1) <= fall_prob:  #Consider fall down probability
                    
                    fall_direction = np.random.randint(1,4,size=1)  #Randomize fall down direction: 1-north, 2-east, 3-south, 4-west
                    
                    if fall_direction == 1 and j-1 >= 0:
                        obstacle_map[i][j-1] = 1
                        if flammable_map[i][j-1] != 0 and fire_map[i][j-1] == 0:
                            fire_map[i][j-1] = int_burning_tensity 
                            
                        
                    elif fall_direction == 2 and i+1 <= map_l-1:
                        obstacle_map[i+1][j] = 1
                        if flammable_map[i+1][j] != 0 and fire_map[i+1][j] == 0:
                            fire_map[i+1][j] = int_burning_tensity 
                        
                    elif fall_direction == 3 and j+1 <= map_h-1:
                        obstacle_map[i][j+1] = 1
                        if flammable_map[i][j+1] != 0 and fire_map[i][j+1] == 0:
                            fire_map[i][j+1] = int_burning_tensity

                                                
                    elif fall_direction == 4 and i-1 >= 0:
                        obstacle_map[i-1][j] = 1
                        if flammable_map[i-1][j] != 0 and fire_map[i-1][j] == 0:
                            fire_map[i-1][j] = int_burning_tensity
                            
            

                
    def print_upate_round(round_count):
        
        round_count += 1

        print('The map has been updated for',str(round_count),'rounds')                     
                        
