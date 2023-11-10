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
        self.flat_size = batch_size
        self.fc0 = nn.Linear(1,GOAL_REPR_SIZE)
        self.fc1 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc2 = nn.Linear(RNN_SIZE, RNN_SIZE)
        self.fc3 = nn.Linear(RNN_SIZE, RNN_SIZE)
        # LSTM cell
        self.lstm= nn.LSTM(RNN_SIZE, RNN_SIZE)
        # Policy and value head
        self.fc_policy = nn.Linear(RNN_SIZE, a_size)
        self.fc_value = nn.Linear(RNN_SIZE, 1)
        
        # Define initial and current LSTM states
        c_init = torch.zeros(1, RNN_SIZE, dtype=torch.float32)
        h_init = torch.zeros(1, RNN_SIZE, dtype=torch.float32)
        self.state_init = [c_init, h_init]
        self.state_in = (torch.zeros(1, RNN_SIZE, dtype=torch.float32), torch.zeros(1, RNN_SIZE, dtype=torch.float32)) # (cell_state, hidden_state)

        self.policy, self.value, self.state_out, _ = self._build_net(self.inputs, self.water_res, a_size)

        if TRAINING:
            self.actions = torch.zeros(batch_size, dtype=torch.int64)
            # print(self.actions.shape)
            self.actions_onehot = F.one_hot(self.actions, a_size).type(torch.float32)
            # print(self.actions_onehot.shape)
            self.target_v = torch.zeros(batch_size, dtype=torch.float32)
            self.advantages = torch.zeros(batch_size, dtype=torch.float32)
            self.responsible_outputs = torch.sum(self.policy * self.actions_onehot, dim=1)

            # Loss Functions
            self.value_loss = 0.5 * torch.sum((self.target_v - self.value.view(-1))**2)
            self.entropy = -0.01 * torch.sum(self.policy * torch.log(torch.clamp(self.policy, 1e-10, 1.0)))
            self.policy_loss = -torch.sum(torch.log(torch.clamp(self.responsible_outputs, 1e-15, 1.0)) * self.advantages)
            self.loss = self.value_loss + self.policy_loss - self.entropy

            # Get gradients from local network using local losses and
            # normalize the gradients using clipping
            trainable_vars = list(self.parameters())
            self.gradients = torch.autograd.grad(self.loss, trainable_vars, create_graph=True)
            self.var_norms = torch.norm(torch.cat([v.view(-1) for v in trainable_vars]))
            self.grad_norms = torch.nn.utils.clip_grad_norm_(self.gradients, GRAD_CLIP)
            self.apply_grads = trainer(self.parameters(), lr=learning_rate)

        print("QAQ! The network is working!")

    def _build_net(self, inputs, water_res, a_size):
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

        # rnn_in = self.h3.unsqueeze(0)
        # TODO: Modify time step HERE.
        sequence_length = 1
        rnn_in = self.h3.unsqueeze(0).unsqueeze(0).expand(sequence_length, -1, -1)
        lstm_out, lstm_state = self.lstm(rnn_in)
        lstm_c, lstm_h = lstm_state
        state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = lstm_out.view(-1, RNN_SIZE)

        policy_layer = self.fc_policy(rnn_out)
        policy = F.softmax(policy_layer, dim=1)
        policy_sig = torch.sigmoid(policy_layer)
        value = self.fc_value(rnn_out)

        return policy, value, state_out, policy_sig
    
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