import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# parameters for training
GRAD_CLIP              = 30.0 # to clip gradients shown by tensorboard at 900 (30^2)
RNN_SIZE               = 128
GOAL_REPR_SIZE         = 12


class ACNet(nn.Module):
    def __init__(self, a_size, batch_size, trainer, learning_rate, TRAINING, GRID_SIZE):
        super(ACNet, self).__init__()
        self.inputs = torch.zeros(batch_size, 4, GRID_SIZE, GRID_SIZE)
        self.water_res = torch.zeros(batch_size, 1)
        self.a_size = a_size
        self.batch_size = batch_size
        self.trainer = trainer
        self.learning_rate = learning_rate
        self.policy = None
        global A_SIZE 
        A_SIZE = a_size

        # Define ACNet layers
        # 4 maps for each agent
        self.conv1 = nn.Conv2d(4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.conv1a = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 2, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(RNN_SIZE // 2, RNN_SIZE - GOAL_REPR_SIZE, kernel_size=2, stride=1)
        self.flat_size = (RNN_SIZE - GOAL_REPR_SIZE) * GRID_SIZE * GRID_SIZE  # Update this size based on your GRID_SIZE
        # self.flat_size = batch_size
        self.fc0 = nn.Linear(1,GOAL_REPR_SIZE)
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc3 = nn.Linear(RNN_SIZE, RNN_SIZE)
        # LSTM
        self.lstm= nn.LSTM(RNN_SIZE, RNN_SIZE, batch_first=True)
        # Policy and value head
        self.fc_policy = nn.Linear(RNN_SIZE, a_size)
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

        print("QAQ! The network is working!")

    # def _build_net(self, inputs, water_res, a_size):
    def forward(self, inputs, water_res):
        a = inputs
        print(inputs.dtype)
        test = self.conv1(inputs)
        conv1 = F.relu(self.conv1(inputs))
        conv1a = F.relu(self.conv1a(conv1))
        conv1b = F.relu(self.conv1b(conv1a))
        pool1 = self.pool1(conv1b)

        conv2 = F.relu(self.conv2(pool1))
        conv2a = F.relu(self.conv2a(conv2))
        conv2b = F.relu(self.conv2b(conv2a))
        pool2 = self.pool2(conv2b)

        conv3 = self.conv3(pool2)

        flat = torch.flatten(conv3, 1)
        water_layer = F.relu(self.fc0(water_res))
        hidden_input = torch.cat([flat, water_layer], dim=1)

        h1 = F.relu(self.fc1(hidden_input))
        h2 = F.relu(self.fc2(h1))
        self.h3 = F.relu(self.fc3(h2 + hidden_input))
        print(self.h3.size())

        rnn_in = self.h3.unsqueeze(0)
        # TODO: Modify time step HERE.
        # sequence_length = 1
        # rnn_in = self.h3.unsqueeze(0).unsqueeze(0).expand(sequence_length, -1, -1)
        lstm_out, lstm_state = self.lstm(rnn_in)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        # self.state_out = (lstm_state[0][:1, :], lstm_state[1][:1, :])
        rnn_out = lstm_out.view(-1, RNN_SIZE)

        policy_layer = self.fc_policy(rnn_out)
        self.policy = F.softmax(policy_layer, dim=1)
        policy_sig = torch.sigmoid(policy_layer)
        self.value = self.fc_value(rnn_out)

        return self.policy, self.value, self.state_out, policy_sig

class ACNetLoss(nn.Module):
    def __init__(self):
        super(ACNetLoss, self).__init__()

    # loss
    # def forward(self, policy, value): 
    def forward(self, policy, value, actions, target_v, advantages):
        # 以下的初始化应该不是必要的，对应的是Tensorlow中的tf.placeholder()操作
        # self.actions = torch.zeros(BATCH_SIZE, dtype=torch.int64)
        # self.target_v = torch.zeros(BATCH_SIZE, dtype=torch.float32)# 这样的初始化应该不是必要的，对应的是Tensorlow中的tf.placeholder()操作
        # self.advantages = torch.zeros(BATCH_SIZE, dtype=torch.float32)# 这样的初始化应该不是必要的，对应的是Tensorlow中的tf.placeholder()操作
        # self.actions_onehot = F.one_hot(self.actions, self.a_size).type(torch.float32)
        self.actions_onehot = F.one_hot(actions, A_SIZE).type(torch.float32)
        self.responsible_outputs = torch.sum(policy * self.actions_onehot, dim=1)
        # 困惑的是，actions, target_v, advantages如何更新在原始TensorFlow代码中并不明确

        # Loss Functions
        # self.value_loss = 0.5 * torch.sum((self.target_v - value.view(-1))**2)
        self.value_loss = 0.5 * torch.sum((target_v - value.view(-1))**2)
        self.entropy = -0.01 * torch.sum(policy * torch.log(torch.clamp(policy, 1e-10, 1.0)))
        # self.policy_loss = -torch.sum(torch.log(torch.clamp(self.responsible_outputs, 1e-15, 1.0)) * self.advantages)
        self.policy_loss = -torch.sum(torch.log(torch.clamp(self.responsible_outputs, 1e-15, 1.0)) * advantages)
        self.total_loss = self.value_loss + self.policy_loss - self.entropy

        # 以下步骤应该是在learning_agent当中使用
        # Get gradients from local network using local losses and
        # normalize the gradients using clipping
        # trainable_vars = list(self.parameters())
        # self.gradients = torch.autograd.grad(self.total_loss, trainable_vars, create_graph=True)
        # self.var_norms = torch.norm(torch.cat([v.view(-1) for v in trainable_vars]))
        # self.grad_norms = torch.nn.utils.clip_grad_norm_(self.gradients, GRAD_CLIP)
        # self.apply_grads = self.trainer(self.parameters(), lr=self.learning_rate)
        
    
# Function for weights initialization
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