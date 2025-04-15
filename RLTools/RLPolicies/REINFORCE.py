import torch

class DiscreteREINFORCE(torch.nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, state):
        return self.net(torch.tensor(state, dtype=torch.float32))
    
    def sample(self, state):
        with torch.no_grad():
            try:
                return self(state).multinomial(num_samples=1, replacement=True).item() 
            except:
                print(self(state))
                raise Exception()

    def sample_training(self, state):
        probs = self(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def reward(self, log_probs, rewards, gamma=0.99):
        loss = 0
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            loss -= log_probs[t] * G  # REINFORCE
        return loss
            
    def sample_best(self,state):
        with torch.no_grad():
            try:
                # print(self(state))
                return torch.argmax(self(state)).item()
            except:
                print(self(state))
                raise Exception()