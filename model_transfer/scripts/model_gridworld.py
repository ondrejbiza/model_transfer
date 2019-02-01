from ..gridworld import GridWorld
from ..model import Model


env = GridWorld()
model = Model(env)
model.start_session()

for i in range(100000):

    if i % 1000 == 0:
        model.show_feature_space()

    model.train_step()

model.stop_session()
