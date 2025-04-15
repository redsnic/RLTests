import torch
from copy import deepcopy

def soft_update(target, source, tau=0.001):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

class DDPG(torch.nn.Module):

    def __init__(self, q_net, p_net, device='cpu'):
        super().__init__()
        self.q_net = q_net.to(device)
        self.p_net = p_net.to(device)
        self.targ_p_net = deepcopy(p_net)
        self.targ_q_net = deepcopy(q_net)
        self.device = device
    
    def q(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q_net(sa)
    
    def tq(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.targ_q_net(sa)
    
    def p(self, state):
        return self.p_net(state)
    
    def tp(self, state):
        return self.targ_p_net(state)

    def _sample(self, state):
        return self.p_net(state)

    def sample(self, state, training=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if training:
            return self._sample(state)
        with torch.no_grad():
            return self._sample(state)

    def q_loss(self, state, action, new_state, reward, is_terminal, gamma=0.99): # reward is computed after taking the action
        return (self.q(state, action) - (reward + (1-is_terminal)*gamma*self.tq(new_state, self.tp(new_state))))**2
    
    def p_loss(self, state, action, new_state, reward, is_terminal, gamma=0.99):
        return -self.q(state, self.p(state))
    
    def freeze_q(self):
        # freeze all parameters of a neural network
        for param in self.q_net.parameters():
            param.requires_grad = False
        for param in self.targ_p_net.parameters():
            param.requires_grad = False
        for param in self.targ_q_net.parameters():
            param.requires_grad = False

    def unfreeze_q(self):
        # unfreeze all parameters of a neural network
        for param in self.q_net.parameters():
            param.requires_grad = True
        for param in self.targ_p_net.parameters():
            param.requires_grad = True
        for param in self.targ_q_net.parameters():
            param.requires_grad = True
    
    def freeze_p(self):
        # freeze all parameters of a neural network
        for param in self.p_net.parameters():
            param.requires_grad = False

    def unfreeze_p(self):
        # unfreeze all parameters of a neural network
        for param in self.p_net.parameters():
            param.requires_grad = True

    def soft_update(self):
        soft_update(self.targ_p_net, self.p_net)
        soft_update(self.targ_q_net, self.q_net)
