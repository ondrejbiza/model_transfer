import matplotlib.pyplot as plt
from ..gridworld import GridWorld
from ..model import Model


env = GridWorld()
model = Model(env, alpha=0.01, learning_rate=0.1, discount=0.9)
model.start_session()

losses = []

for i in range(1001):

    if i == 300 or i == 700 or i == 1000:

        print("step {:d}".format(i))
        print(model.policy_evaluation(env.uniform_policy, env.uniform_policy_values))
        model.show_feature_space()

    loss = model.train_step()
    losses.append(loss)

plt.semilogy(range(1, len(losses) + 1), losses)
plt.show()

model.stop_session()
