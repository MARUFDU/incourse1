import numpy as np
import matplotlib.pyplot as plt

def r(t):
    return np.array([5*np.cos(t),4*np.sin(t)])

def v(t):
    return np.array([-5*np.sin(t),4*np.cos(t)])

t1=np.pi/4
t2=np.pi

r_t1=r(t1)
v_t1=v(t1)

r_t2=r(t2)
v_t2=v(t2)

t_values=np.linspace(0,2*np.pi,400)
curve=[]
for t in t_values:
    curve.append(r(t))

curve=np.array(curve)

plt.plot(curve[:,0],curve[:,1],label='Curve C: r(t)=(5cos(t),4sin(t))')

plt.quiver(0,0,r_t1[0],r_t1[1],angles='xy',scale_units='xy',scale=1,color='blue',label='r(pi/4)')
plt.quiver(0,0,r_t2[0],r_t2[1],angles='xy',scale_units='xy',scale=1,color='green',label='r(pi)')

plt.quiver(r_t1[0],r_t1[1],v_t1[0],v_t1[1],angles='xy',scale_units='xy',scale=1,color='cyan',label='r(pi/4)')
plt.quiver(r_t2[0],r_t2[1],v_t2[0],v_t2[1],angles='xy',scale_units='xy',scale=1,color='purple',label='r(pi)')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
