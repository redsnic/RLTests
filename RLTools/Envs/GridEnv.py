import torch 
import gymnasium as gym

class GridEnv(gym.Env):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(size*size)
        self.state = None
        self.reset()
        self.obstacles = []
        self.max_steps = 10
        self.current_step = 0

    def reset(self, state=None): # observation, info
        # self.state = (0, 0)
        # random initial state 
        self.state = (torch.randint(self.size-1, (1,)).item(), torch.randint(self.size-1, (1,)).item())
        if state is not None:
            self.state = state
        self.current_step = 0
        return self.state, {}

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        for obs in obstacles:
            if obs[0] < 0 or obs[0] >= self.size or obs[1] < 0 or obs[1] >= self.size:
                raise ValueError("Obstacle coordinates out of bounds.")
            if obs == (self.size - 1, self.size - 1):
                raise ValueError("Obstacle cannot be at the goal position.")

    def render(self, mode='human'):
        grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        # if in bounds
        if not(self.state[0] < 0 or self.state[0] >= self.size or self.state[1] < 0 or self.state[1] >= self.size):
            grid[self.state[0]][self.state[1]] = 'A'
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        grid[self.size - 1][self.size - 1] = 'G'
        if mode == 'human':
            print('\n'.join(['.'.join(row) for row in grid]))
            print()
        return grid

    def translate_action_to_human(self, action):
        action_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        return action_dict.get(action, "Invalid action")

    def step(self, action):
        x, y = self.state
        if action == 0:  # up
            x = x - 1
        elif action == 1:  # down
            x = x + 1
        elif action == 2:  # left
            y = y - 1
        elif action == 3:  # right
            y = y + 1

        self.state = (x, y)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            return self.state, -0., False, True, {}

        if (x, y) == (self.size - 1, self.size - 1): # goal
            # print("goal")
            return self.state, 1000.0, True, False, {}
        else:
            for obs in self.obstacles: # wall
                if (x, y) == obs:
                    # print("obstacle")
                    return self.state, -1.0*(self.max_steps-self.current_step), True, False, {}
            # if our of bounds
            if x < 0 or x >= self.size or y < 0 or y >= self.size:
                # print("out of bounds")
                return self.state, -1.0*(self.max_steps-self.current_step), True, False, {}
        # If the agent moves to a valid position
        return self.state, -0.01, False, False, {}
