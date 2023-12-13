import matplotlib.pyplot as plt
import numpy as np
with open('./trained_data/base/score.data','r') as f:
    numbers = f.readlines()
# 绘制折线图
numbers = np.array([max(eval(i.replace('\n','')),0) for i in numbers])
plt.plot(numbers)
plt.yticks([i for i in range(0,30,2)])
# 添加标题和轴标签
plt.xlabel("episode")
plt.ylabel("Score")
# 显示图表
plt.show()
plt.savefig("score.png")
