import random

a = [i for i in range(1, 7)]

for _ in range(6):
    print(random.sample(a, 6))