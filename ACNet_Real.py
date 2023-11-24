import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
# 
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# parameters for training
GRAD_CLIP              = 30.0 # to clip gradients shown by tensorboard at 900 (30^2)
RNN_SIZE               = 128
GOAL_REPR_SIZE         = 12

Transition = namedtuple('Transition',
                        ('state_map', 'state_water', 'action', 'next_state_map', 'next_state_water', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.constant_(param.data, 0)

class ACNet(nn.Module):

    def __init__(self, n_actions, GRID_SIZE):
        super(ACNet, self).__init__()
        global A_SIZE 
        A_SIZE = n_actions

        # Define ACNet layers
        # 4 maps for each agent
        self.conv1 = nn.Conv2d(4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, kernel_size=3, stride=1,padding=1)
        self.conv2b = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE - GOAL_REPR_SIZE, kernel_size=3, stride=1)
        self.flat_size = (RNN_SIZE - GOAL_REPR_SIZE) * GRID_SIZE * GRID_SIZE  # Update this size based on your GRID_SIZE
        # self.flat_size = batch_size
        self.fc0 = nn.Linear(1,GOAL_REPR_SIZE)
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc3 = nn.Linear(RNN_SIZE, RNN_SIZE)
        # LSTM
        self.lstm= nn.LSTM(RNN_SIZE, RNN_SIZE, batch_first=True)
        # Policy and value head
        self.fc_policy = nn.Linear(RNN_SIZE, n_actions)
        self.fc_value = nn.Linear(RNN_SIZE, 1)
        
        # Define initial and current LSTM states
        c_init = torch.zeros(1, RNN_SIZE, dtype=torch.float32)
        h_init = torch.zeros(1, RNN_SIZE, dtype=torch.float32)
        self.state_init = [c_init, h_init]
        self.state_in = (torch.zeros(1, RNN_SIZE, dtype=torch.float32), torch.zeros(1, RNN_SIZE, dtype=torch.float32)) # (cell_state, hidden_state)

        # self.policy, self.value, self.state_out, _ = self._build_net(self.inputs, self.water_res, a_size)
        # self.policy, self.value, self.state_out, _ = self._build_net(self.inputs, self.water_res)

        # Initialize model weights
        self.apply(weights_init)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input, water_res):
        # print(state)
        #print(input.shape)
        conv1 = F.relu(self.conv1(input))
        conv1a = F.relu(self.conv1a(conv1))
        conv1b = F.relu(self.conv1b(conv1a))
        # pool1 = self.pool1(conv1b)

        conv2 = F.relu(self.conv2(conv1b))
        conv2a = F.relu(self.conv2a(conv2))
        conv2b = F.relu(self.conv2b(conv2a))
        # pool2 = self.pool2(conv2b)

        conv3 = self.conv3(conv2b)
        # print(conv3.shape)
        # flattened_size = conv3_shape[1] * conv3_shape[2] * conv3_shape[3]
        # print(conv3)
        flat = conv3.view(-1, RNN_SIZE - GOAL_REPR_SIZE)
        # print(flat) 
        water_layer = F.relu(self.fc0(water_res))
        
        # water_layer = torch.transpose(water_layer, 0, 1)
        #  flat = torch.transpose(flat, 0, 1)
        # print("flat")
        # print(flat.shape)
        # print("water")
        # print(water_layer.shape)
        hidden_input = torch.cat([flat, water_layer], dim = 1)
        # print(hidden_input.shape)

        h1 = F.relu(self.fc1(hidden_input))
        h2 = F.relu(self.fc2(h1)) 
        self.h3 = F.relu(self.fc3(h2 + hidden_input))
        # print(self.h3.size())

        rnn_in = self.h3.unsqueeze(0)
        # TODO: Modify time step HERE.
        # sequence_length = 1
        # rnn_in = self.h3.unsqueeze(0).unsqueeze(0).expand(sequence_length, -1, -1)
        lstm_out, lstm_state = self.lstm(rnn_in)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        # self.state_out = (lstm_state[0][:1, :], lstm_state[1][:1, :])
        rnn_out = lstm_out.view(-1, RNN_SIZE)

        self.policy = self.fc_policy(rnn_out)
        self.policy = F.softmax(self.policy, dim=1)
        
        self.value = self.fc_value(rnn_out)
        
        # Uncomment the above if use L1 loss
        return self.policy, self.value
    



