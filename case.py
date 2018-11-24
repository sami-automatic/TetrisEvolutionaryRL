from engineLittle import TetrisEngine

env = TetrisEngine(10,20)

obs = env.clear()

obs, reward, done= env.step(2)
print(reward)
obs, reward, done= env.step(2)
print(reward)
obs, reward, done= env.step(2)
print(reward)
obs, reward, done= env.step(2)
print(reward)

print(obs)

print(reward)

