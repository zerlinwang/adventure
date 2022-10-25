from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# 环境初始化
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 设置动作空间
env = JoypadSpace(env, SIMPLE_MOVEMENT)

import pdb
pdb.set_trace()
done = True
# 与环境交互5000步
for step in range(5000):
    # 如果游戏结束，那么重置游戏
    if done:
        state = env.reset()
    # 从mario可选的动作中随机采样一个动作
    action = env.action_space.sample()
    # 将动作给到环境，环境返回下一帧的状态、这一帧获得的奖励、游戏是否结束以及其它的信息
    state, reward, done, info = env.step(action)
    # 环境渲染当前一帧内容
    env.render()
# 关闭环境
env.close()
