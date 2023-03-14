import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库

    
class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.nrow - 1  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:  # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x
    
class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
    
ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 1
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma, n_action=4)#初始化Sara算法
num_episodes = 500  # 智能体在环境中运行的序列的数量

return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:#未到达终点
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.figure(1)
plt.plot(episodes_list, return_list,label='Sarsa',c='r')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Cliff Walking')
plt.figure(2)
plt.plot(episodes_list, return_list,c='r')
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()



        
def print_route(agent, env, action_meaning,  disaster=[],end=[]):
    state = env.reset()
    done=False
    action = agent.take_action(state)
    state_list=[]
    action_list=[]
    num_action=0
    while not done:
        state_list.append(state)
        action_list.append(action)
        next_state, reward, done = env.step(action)
        next_action = agent.take_action(next_state)
        agent.update(state, action, reward, next_state, next_action)
        num_action+=1
        state = next_state
        action = next_action
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('**', end=' ')
            elif (i * env.ncol + j) in end:
                print('EE', end=' ')
            elif (i * env.ncol + j) in state_list:
                pi_str = ''
                a=agent.best_action(i * env.ncol + j)
                for k in range(len(action_meaning)):
                    if a[k] > 0:    
                        pi_str=action_meaning[k]
                if pi_str=='':
                    pi_str='oo' 
                print(pi_str, end=' ')
            else:
                pi_str = 'oo'
                print(pi_str, end=' ')
        print()
    return num_action  
        
        

action_meaning = ['^^', 'vv', '<<', '>>']
print('Sarsa算法最终收敛得到的路径为：')
num_action=print_route(agent, env, action_meaning , list(range(37, 47)), [47])
print('Sarsa算法最终收敛得到的步数为：%d'%num_action)
print('Sarsa算法最终收敛得到的策略为：')
action_meaning = ['^', 'v', '<', '>']
print_agent(agent, env, action_meaning ,list(range(37, 47)), [47])


class QLearning:
    """ Q-learning算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def take_action(self, state):  #选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max(
        ) - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error
        
    
ncol = 12
nrow = 4
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 1
agent = QLearning(ncol, nrow, epsilon, alpha, gamma)#初始化Q-learning算法


return_list = []  # 记录每一条序列的回报
for i in range(10):  # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state)
                state = next_state
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.figure(1)
plt.plot(episodes_list, return_list,label='Q-learning',c='b')
plt.legend()
plt.figure(3)
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on {}'.format('Cliff Walking'))



def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()



        
def print_route(agent, env, action_meaning,  disaster=[],end=[]):
    state = env.reset()
    done=False
    action = agent.take_action(state)
    state_list=[]
    action_list=[]
    num_action=0
    while not done:
        action_list.append(action)
        state_list.append(state)
        action = agent.take_action(state)
        next_state, reward, done = env.step(action) 
        agent.update(state, action, reward, next_state)
        state = next_state
        num_action+=1
        if num_action>100:
            break
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('**', end=' ')
            elif (i * env.ncol + j) in end:
                print('EE', end=' ')
            elif (i * env.ncol + j) in state_list:
                pi_str = ''
                a=agent.best_action(i * env.ncol + j)
                for k in range(len(action_meaning)):
                    if a[k] > 0:    
                        pi_str=action_meaning[k]
                if pi_str=='':
                    pi_str='oo' 
                print(pi_str, end=' ')
            else:
                pi_str = 'oo'
                print(pi_str, end=' ')
        print()
    return num_action  
        
        

action_meaning = ['^^', 'vv', '<<', '>>']
print('Q-learning算法最终收敛得到的路径为：')
num_action=print_route(agent, env, action_meaning , list(range(37, 47)), [47])
print('Q-learning算法最终收敛得到的步数为：%d'%num_action)
print('Q-learning算法最终收敛得到的策略为：')
action_meaning = ['^', 'v', '<', '>']
print_agent(agent, env, action_meaning ,list(range(37, 47)), [47])



plt.show()