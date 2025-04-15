import random

class ReplayBuffer():

    def __init__(self):
        self.events = []
        self.sweeping_index = 0

    def add(self, states, actions, rewards):
            if len(self.events) < 100000:
                for i in range(len(states)):
                    if i == len(states)-1: # terminal
                        self.events.append([ # state, action,rewads, 
                            states[i],
                            actions[i],
                            states[i],
                            rewards[i],
                            float((i == len(states)-1)) # reaching terminal ?
                        ])
                    else:
                        # print(states[i].shape, len(actions), i)
                        self.events.append([ # state, action,rewads, 
                            states[i],
                            actions[i],
                            states[i+1],
                            rewards[i],
                            False # reaching terminal ?
                        ])
            else: # replace older samples
                for i in range(len(states)):
                    if i == len(states)-1: # terminal
                        self.events[self.sweeping_index] = [ # state, action,rewads, 
                            states[i],
                            actions[i],
                            states[i],
                            rewards[i],
                            float((i == len(states)-1)) # reaching terminal ?
                        ]
                    else:
                        # print(states[i].shape, len(actions), i)
                        self.events[self.sweeping_index] = [ # state, action,rewads, 
                            states[i],
                            actions[i],
                            states[i+1],
                            rewards[i],
                            False # reaching terminal ?
                        ]
                self.sweeping_index += 1
                if self.sweeping_index == len(self.events):
                    self.sweeping_index = 0

    def sample(self, n):
        return random.choices(self.events, k=n) 
