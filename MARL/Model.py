import torch
import torch.nn as nn
import torch.nn.functional as F


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()

    def act(self, input):
        policy, value = self.forward(input)
        return policy, value


class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.linear_c1 = nn.Linear(action_shape_n + obs_shape_n, args.num_units_openai0)
        self.linear_c2 = nn.Linear(args.num_units_openai0, args.num_units_openai1)
        self.linear_c3 = nn.Linear(args.num_units_openai1, args.num_units_openai2)
        self.linear_c4 = nn.Linear(args.num_units_openai2, args.num_units_openai3)
        self.linear_c = nn.Linear(args.num_units_openai3, 1)
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        x = self.LReLU(self.linear_c3(x))
        x = self.LReLU(self.linear_c4(x))
        value = self.linear_c(x)
        return value


class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(openai_actor, self).__init__()
        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai0)
        self.linear_a2 = nn.Linear(args.num_units_openai0, args.num_units_openai1)
        self.linear_a3 = nn.Linear(args.num_units_openai1, args.num_units_openai2)
        self.linear_a4 = nn.Linear(args.num_units_openai2, args.num_units_openai3)
        self.linear_a = nn.Linear(args.num_units_openai3, action_size)
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, input, model_original_out=False):
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        x = self.LReLU(self.linear_a3(x))
        x = self.LReLU(self.linear_a4(x))
        model_out = self.linear_a(x)
        # policy = self.tanh(model_out)
        # policy = (u*180+360) % 360
        # u = torch.rand_like(model_out)

        u = torch.rand(model_out.size(), dtype=model_out.dtype, layout=model_out.layout, device=model_out.device)

        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out:
            return model_out, policy
        return policy
