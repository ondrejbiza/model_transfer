from ..gridworld import GridWorld
from ..model import Model


env = GridWorld()
model = Model(env)
model.start_session()

for i in range(200000):

    if i % 40000 == 0:

        if i > 0:
            model.k_means_update()

        model.show_feature_space()

    model.train_step()

model.stop_session()
