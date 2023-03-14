import numpy as np
import copy
class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env#环境
        self.v = np.zeros([self.env.nrow,self.env.ncol]) # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [[None for i in range(self.env.nrow)] for j in range(self.env.ncol)]
    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = np.zeros([self.env.ncol,self.env.nrow])#新的状态价值列表
            for row in range(self.env.nrow): 
                for col in range(self.env.ncol):
                    qsa_list = []  # 开始计算状态(col,row)下的所有Q(s,a)价值
                    for a in range(4):
                        r,next_state,p_ssa = env.query(col, row, a)#奖励，下一个状态，状态转移概率
                        qsa = r + self.gamma * self.v[next_state] * p_ssa#动作价值
                        qsa_list.append(qsa)  #动作价值列表# 这一行和下一行代码是价值迭代和策略迭代的主要区别
                    new_v[row,col] = max(qsa_list)#选最大的作为当前位置的状态价值
                    max_diff = max(max_diff, abs(new_v[row,col] - self.v[row,col]))#迭代的差
            self.v = new_v#更新状态价值
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt += 1
            print("第%d轮迭代结果"%cnt)
            self.get_policy()
            print_agent(self, action_meaning,[[0,0],[my_row-1,my_col-1]])
        print("价值迭代一共进行%d轮,最终结果为:" % cnt)
        self.get_policy()

    def get_policy(self):  # 根据价值函数导出一个贪婪策略
        for row in range(self.env.nrow): 
            for col in range(self.env.ncol):#每个位置都遍历一下
                qsa_list = []
                for a in range(4):
                    r,next_state,p_ssa = env.query(col, row, a)
                    qsa = r + self.gamma * self.v[next_state] * p_ssa
                    qsa_list.append(qsa)
                maxq = max(qsa_list)#选动作价值最大的作为当前位置的策略
                cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
                # 让这些动作均分概率
                self.pi[row][col] = [1 / cntq if q == maxq else 0 for q in qsa_list]
                
                
class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env#环境
        self.v = np.zeros([self.env.nrow,self.env.ncol]) # 初始化价值为0
        self.pi = [[[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.ncol)] for j in range(self.env.nrow)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 0  # 计数器
        while 1:
            cnt+=1
            max_diff = 0
            new_v = np.zeros([self.env.nrow,self.env.ncol])#新的状态评价函数
            for row in range(self.env.nrow):
                for col in range(self.env.ncol):
                    qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值，这个数组存储四个动作的Q(s,a)值
                    for a in range(4):
                        r,next_state,p_ssa = env.query(col, row, a)#回报，下一个状态，状态转移概率
                        qsa = r + self.gamma * self.v[next_state] * p_ssa
                        qsa_list.append(self.pi[row][col][a] * qsa)#pi为策略概率和奖励相乘
                    new_v[row][col] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                    max_diff = max(max_diff, abs(new_v[row][col]- self.v[row][col]))
            self.v = new_v#更新状态价值
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代

    def policy_improvement(self):  # 策略提升
        for row in range(self.env.nrow): 
            for col in range(self.env.ncol):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for a in range(4):
                        r,next_state,p_ssa = env.query(col, row, a)
                        qsa = r + self.gamma * self.v[next_state] * p_ssa
                        qsa_list.append(qsa)
                maxq = max(qsa_list)
                cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
                # 让这些动作均分概率
                self.pi[row][col] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        #print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        cnt=0
        while 1:
            cnt+=1
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            new_pi = self.policy_improvement()
            print("第%d轮策略迭代结果" % cnt) 
            print_agent(self,['<','v','>','^'],[[0,0],[my_row-1,my_col-1]])
            #if old_pi == new_pi:
            if cnt==10:   break


class Env:
    def __init__(self,nrow,ncol):
        self.nrow = nrow
        self.ncol=ncol
        self.P=-1*np.ones([nrow,ncol])
        self.P[0][0]=0
        self.P[nrow-1][ncol-1]=0
    def query(self,incol,inrow,a):
        if a==0:
            col=incol-1
            row=inrow
        elif a==1:
            col=incol
            row=inrow+1
        elif a==2:
            col=incol+1
            row=inrow
        elif a==3:
            col=incol
            row=inrow-1
        if col<0 or col>=self.ncol or row<0 or row>=self.nrow:#越界情况位置保持不变
            col=incol
            row=inrow
        return self.P[row][col],(row,col),1


def print_agent(agent, action_meaning,end=[]):#输入需要打印的agent，每个数字（0到3）代表的对应动作，还有终止态的位置
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i,j]), end=' ')
        print()
    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if ([i,j]) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i][j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()
    
    
        
if __name__=='__main__':
    my_row=4
    my_col=4
    env = Env(my_row,my_col)
    action_meaning =['<','v','>','^'] 
    theta = 0.001#阈值，策略的状态价值小于这个值退出迭代
    gamma = 0.9
    flag=int(input("选择进行策略迭代(输入0)还是价值迭代(输入1):"))
    if flag==1:
        agent = ValueIteration(env, theta, gamma)
        agent.value_iteration()
        print_agent(agent, action_meaning,[[0,0],[my_row-1,my_col-1]])
    if flag==0:
        agent = PolicyIteration(env, theta, gamma)
        agent.policy_iteration()
        print("最终的状态价值函数：")
        print_agent(agent, action_meaning,[[0,0],[my_row-1,my_col-1]])