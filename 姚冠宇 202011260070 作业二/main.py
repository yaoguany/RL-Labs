import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # tqdm是显示循环进度条的库

def wind(x,y):#风
    if x in [3,4,5,8]:
        return y-1
    elif x in [6,7]:
        return y-2
    else:
        return y
    
    
class CliffWalkingEnv:#环境
    def __init__(self, ncol=10, nrow=7):
        self.nrow = nrow
        self.ncol = ncol
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = 3  # 记录当前智能体位置的纵坐标

    def step(self, action):  # 外部调用这个函数来改变当前位置
        # 8种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右，change[4]:左上，change[5]:右下，change[6]:左下，change[7]:右上。坐标系原点(0,0).
        # 定义在左上角'<^','v>','<v','^>'
        change = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1,-1],[1,1],[-1,1],[1,-1]]
        self.y=wind(self.x,self.y)#风
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        #self.y=wind(self.x,self.y)
        next_state = self.y * self.ncol + self.x#下一个状态
        reward = -1#奖励
        done = False#是否结束
        if self.y == 3 and self.x == 7:  # 下一个位置在目标
            done = True
            reward=0
        return next_state, reward, done

    def reset(self):  # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = 3
        return self.y * self.ncol + self.x
    
class Sarsa:
    """ Sarsa算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=8):
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
        Q_max = np.max(self.Q_table[state])#当前位置选择的动作
        a = [0 for _ in range(self.n_action)]#初始化动作
        for i in range(self.n_action):  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1):# Sara算法更新Q表格
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]#TD误差
        self.Q_table[s0, a0] += self.alpha * td_error
    
ncol = 10
nrow = 7
env = CliffWalkingEnv(ncol, nrow)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 1
agent = Sarsa(ncol, nrow, epsilon, alpha, gamma, n_action=8)#初始化Sara算法
num_episodes = 600000  # 智能体在环境中运行的序列的数量

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
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))


def print_agent(agent, env, action_meaning, end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in end:
                print('EE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'oo'
                print(pi_str, end=' ')
        print()
    for i in range(env.ncol):
        if i in [3,4,5,8]:
            print('1111111111111111', end=' ')
        elif i in [6,7]:
            print('2222222222222222', end=' ')
        else:
            print('================', end=' ')
    print()
        
def print_route(agent, env, action_meaning, end=37):
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
            if (i * env.ncol + j) == end:
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
    for i in range(env.ncol):
        if i in [3,4,5,8]:
            print('11', end=' ')
        elif i in [6,7]:
            print('22', end=' ')
        else:
            print('==', end=' ')
    print()
    return num_action  
        
        

action_meaning = ['^^', 'vv', '<<', '>>','<^','v>','<v','^>']
print('Sarsa算法最终收敛得到的路径为：')
num_action=print_route(agent, env, action_meaning , 37)
print('Sarsa算法最终收敛得到的步数为：%d'%num_action)
print('Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning , [37])
plt.show()