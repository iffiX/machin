import matplotlib.pyplot as plt

def draw_agent_num_figure(agent_nums):
    f = plt.figure()
    ax = f.add_subplot("111")
    ax.plot([i for i in range(len(agent_nums[0]))],
           agent_nums[0],
           "g^-",
           [i for i in range(len(agent_nums[1]))],
           agent_nums[1],
           "rx-")
    return f