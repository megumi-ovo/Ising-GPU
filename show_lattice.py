from numpy import *
import matplotlib.pyplot as plt

data1 = loadtxt('data1.txt')
data2 = loadtxt('data6.txt')

plt.subplot(1,2,1)
plt.imshow(data1)
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(data2)
plt.axis('off')

plt.show()