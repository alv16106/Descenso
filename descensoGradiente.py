import numpy as np

def cost(x, y, theta):
  m, n = x.shape
  h = np.matmul(x,theta)
  difsq = (y - h) ** 2
  return difsq.sum() / (2 * m)

# FUNCTION FOR COST OF DERIVATE FUNCTION J'
def derivative_cost(x, y, theta):
  m, n = x.shape
  h = np.matmul(x,theta)
  dif = np.dot((h - y), x)
  return dif.sum() / m


def DG(x, y, tetha, alpha, cost, derivative_cost, max_iter = 10000):
  iterations = 0
  derivative = 10000
  current_tetha = tetha
  while((iterations < max_iter) and (derivative != 0)):
    current_cost = cost(x, y, current_tetha)
    current_tetha = current_tetha - (alpha * derivative_cost(x, y, current_tetha))
    derivative = abs(cost(x, y, current_tetha) - current_cost)
    iterations += 1
