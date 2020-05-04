from utils.parallel.assigner import ModelAssigner
from models.naive.env_walker import Actor, Critic

if __name__ == "__main__":
    actors = [Actor(100, 20, 1) for i in range(5)]
    critics = [Critic(100, 20) for i in range(5)]
    ass = ModelAssigner(actors + critics,
                        {(i, i+5): 1 for i in range(5)},
                        model_with_input_size_multiplier=50,
                        devices=["cuda:0", "cpu"])
    print(ass.assignment[:5])
    print(ass.assignment[5:])