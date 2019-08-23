from numpy import *
import matplotlib.pyplot as plt

data = loadtxt('data.txt')

ITER = data.shape[0]
x = linspace(1.0,4.0,num=ITER)

T_c = [2.269,2.269]
x_int = [1.0,4.0]

font = {'family':'serif'}
plt.rc('font',**font)
plt.rc('text',usetex=True)

plt.subplot(2,2,1)
plt.scatter(x,data[:,0],s=0.03)
y_int = [-0.1,1.1]
plt.ylim(y_int)
plt.plot(T_c,y_int,linewidth=0.5,color='red')
plt.xlabel(r'$T$')
plt.ylabel(r'magnetization $M$')

plt.subplot(2,2,2)
plt.scatter(x,data[:,1],s=0.03)
y_int = [0,300]
plt.ylim(y_int)
plt.plot(T_c,y_int,linewidth=0.5,color='red')
plt.xlabel(r'$T$')
plt.ylabel(r'susceptibility $X$')

plt.subplot(2,2,3)
plt.scatter(x,data[:,2],s=0.03)
y_int = [-2.05,-0.55]
plt.ylim(y_int)
plt.plot(T_c,y_int,linewidth=0.5,color='red')
plt.xlabel(r'$T$')
plt.ylabel(r'energy $E$')

plt.subplot(2,2,4)
plt.scatter(x,data[:,3],s=0.03)
y_int = [0,3]
plt.ylim(y_int)
plt.plot(T_c,y_int,linewidth=0.5,color='red')
plt.xlabel(r'$T$')
plt.ylabel(r'specific heat $C$')

plt.show()