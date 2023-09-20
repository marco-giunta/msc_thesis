# Problem: double each number from 0 to 4
# Imperative Programming Solution:
x = []
for n in range(5):
    x.append(n*2)
# Declarative Programming Solution:
def double_number(n):
    return n*2
y = list(map(double_number, range(5)))