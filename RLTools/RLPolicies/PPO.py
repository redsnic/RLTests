
import torch
from copy import deepcopy

class PPO(torch.nn.Module):

    def __init__(self, net, v_net, eps=0.2, device='cpu', entropy_weight=0.01):
        super().__init__()
        self.net = net.to(device)
        self.net_old = deepcopy(net).to(device)
        self.eps = eps
        self.device = device
        self.entropy_weight = entropy_weight
        self.freeze(self.net_old)
        self.v_net = v_net.to(device)

    def forward(self, x):
        return self.net(x)
    
    def advantage(self, states, rewards, gamma=0.99):
        with torch.no_grad():
            values = self.v(states).squeeze(1)
        discounts = gamma ** torch.arange(rewards.shape[0]).to(self.device)
        discounted_rewards = torch.cumsum((rewards.flip(0) * discounts), dim=0).flip(0) / discounts
        advantage = discounted_rewards - values
        # normalize advantage
        # advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
        return advantage, discounted_rewards

    def clipCPI(self, states, actions, advantage):
        risk_ratio = self.p(states, actions) / (self.p_old(states, actions) + 1e-6)
        return torch.min(
            risk_ratio * advantage,
            torch.clip(risk_ratio, 1 - self.eps, 1 + self.eps) * advantage
        )

    def value_loss(self, states, discounted_rewards):
        v_pred = self.v(states).squeeze(1)
        return torch.nn.functional.mse_loss(v_pred, discounted_rewards)

    def entropy_bonus(self, entropy):
        entropy = (entropy).mean()
        return self.entropy_weight * entropy

    def reward(self, states, actions, rewards, entropy, gamma=0.99, value_loss_weight=0.5):

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        entropy = torch.tensor(entropy, dtype=torch.float32).to(self.device)

        advantage, discounted_rewards = self.advantage(states, rewards, gamma)
        cpi = self.clipCPI(states, actions, advantage.detach()).mean()
        entropy_bonus = self.entropy_bonus(entropy)
        v_loss = self.value_loss(states, discounted_rewards)
        total_loss = -(cpi + entropy_bonus) + value_loss_weight * v_loss
        return total_loss

    def swap(self):
        self.net_old.load_state_dict(self.net.state_dict())
        self.freeze(self.net_old)
        self.unfreeze(self.net)
    
    def freeze(self, net):
        # freeze all parameters of a neural network
        for param in net.parameters():
            param.requires_grad = False
    
    def unfreeze(self, net):
        for param in net.parameters():
            param.requires_grad = True

    def sample_training(self, state):
        probs = self.net(torch.tensor(state).to(self.device))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        entropy = dist.entropy()
        return action.item(), dist.log_prob(action), entropy
    
    def sample(self, state):
        with torch.no_grad():
            try:
                return self(state).multinomial(num_samples=1, replacement=True).item() 
            except:
                print(self(state))
                raise Exception()
            
    def sample_best(self, state):
        state = torch.tensor(state).to(self.device)
        with torch.no_grad():
            try:
                # print(self(state))
                return torch.argmax(self(state)).item()
            except:
                print(self(state))
                raise Exception()
            
    def p(self, state, action):
        # print(self.net(state).shape)
        # print(action.shape)
        # print(self.net(state)[action].shape)
        o = self.net(state)
        return o.gather(1, action.unsqueeze(1)).squeeze(1)

    def p_old(self, state, action):
        o = self.net_old(state)
        return o.gather(1, action.unsqueeze(1)).squeeze(1)
    
    def v(self, state):
        o = self.v_net(state)
        return o



import torch
from copy import deepcopy

class BPPO(torch.nn.Module):

    def __init__(self, net, v_net, eps=0.2, device='cpu', entropy_weight=0.01):
        super().__init__()
        self.net = net.to(device)
        self.net_old = deepcopy(net).to(device)
        self.eps = eps
        self.device = device
        self.entropy_weight = entropy_weight
        self.freeze(self.net_old)
        self.v_net = v_net.to(device)

    def forward(self, x):
        return self.net(x)
    
    def advantage(self, states, rewards, mask, gamma=0.99):
        with torch.no_grad():
            values = self.v(states).squeeze(2)
        discounts = gamma ** torch.arange(rewards.shape[1]).to(self.device)
        discounted_rewards = torch.cumsum((rewards.flip(1) * discounts), dim=1).flip(1) / discounts
        advantage = discounted_rewards - values

        # masking
        advantage = advantage * mask
        discounted_rewards = discounted_rewards * mask

        # normalize advantage
        # advantage = (advantage - advantage.mean())/(advantage.std() + 1e-8)
        return advantage, discounted_rewards

    def clipCPI(self, states, actions, advantage, mask):
        risk_ratio = self.p(states, actions) / (self.p_old(states, actions) + 1e-6)
        clip_cpi = torch.min(
            risk_ratio * advantage,
            torch.clip(risk_ratio, 1 - self.eps, 1 + self.eps) * advantage
        )
        return clip_cpi * mask

    def value_loss(self, states, discounted_rewards, mask):
        v_pred = self.v(states).squeeze(2)
        v_pred = v_pred * mask
        return torch.nn.functional.mse_loss(v_pred, discounted_rewards)

    def entropy_bonus(self, entropy, mask):
        entropy = (entropy * mask).mean()
        return self.entropy_weight * entropy

    def reward(self, states, actions, rewards, entropy, mask, gamma=0.99, value_loss_weight=0.5):

        advantage, discounted_rewards = self.advantage(states, rewards, gamma, mask)
        cpi = self.clipCPI(states, actions, advantage.detach(), mask).mean()
        entropy_bonus = self.entropy_bonus(entropy, mask)
        v_loss = self.value_loss(states, discounted_rewards, mask)
        total_loss = -(cpi + entropy_bonus) + value_loss_weight * v_loss
        return total_loss

    def swap(self):
        self.net_old.load_state_dict(self.net.state_dict())
        self.freeze(self.net_old)
        self.unfreeze(self.net)
    
    def freeze(self, net):
        # freeze all parameters of a neural network
        for param in net.parameters():
            param.requires_grad = False
    
    def unfreeze(self, net):
        for param in net.parameters():
            param.requires_grad = True

    def sample_training(self, state):
        probs = self.net(torch.tensor(state).to(self.device))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        entropy = dist.entropy()
        return action, dist.log_prob(action), entropy
    
    def sample(self, state):
        with torch.no_grad():
            try:
                return self(state).multinomial(num_samples=1, replacement=True)
            except:
                print(self(state))
                raise Exception()
            
    def sample_best(self, state):
        state = torch.tensor(state).to(self.device)
        with torch.no_grad():
            try:
                # print(self(state))
                return torch.argmax(self(state))
            except:
                print(self(state))
                raise Exception()
            
    def p(self, state, action):
        # print(self.net(state).shape)
        # print(action.shape)
        # print(self.net(state)[action].shape)
        o = self.net(state)
        return o.gather(2, action.unsqueeze(2)).squeeze(2)

    def p_old(self, state, action):
        o = self.net_old(state)
        return o.gather(2, action.unsqueeze(2)).squeeze(2)
    
    def v(self, state):
        o = self.v_net(state)
        return o
