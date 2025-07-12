import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import sympy as smp

f=lambda z,y,x:x*np.exp(-y)*np.cos(z)
anti_derivative_1=sp.integrate.tplquad(f,0,1, lambda x:0,lambda x:1-x**2,lambda x,y:3,lambda x,y:4-x**2-y**2)[0]
print(f'the values of integral 1 is: {anti_derivative_1}\n')

g=lambda y,x:(x*y)/np.sqrt(x**2+y**2+1)
anti_derivative_2=sp.integrate.dblquad(g,0,1,0,1)[0]
print(f'the values of integral 2 is: {anti_derivative_2}\n')

#b
x,y,z=smp.symbols('x y z',real=True)
z=smp.sqrt(4-x**2)
dzdx=z.diff(x)
dzdy=z.diff(y)
dzdz=1
integrand=smp.sqrt(dzdx**2+dzdy**2+dzdz**2)
integral_1=smp.integrate(integrand,(x,0,1),(y,0,4))
#print(integral_1)
#print(integrand)
integrand_1=smp.lambdify([y,x],integrand)
integral_2=sp.integrate.dblquad(integrand_1,0,1,0,4)[0] 
#y defined first so integral of y is calculated first and then x
#like integral{0-1}integral{0-4}( )dydx
print(f'the surface area is: {integral_2}')

==================================================================

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as smp

x,y,z=smp.symbols('x y z',real=True)
r,theta=smp.symbols('r theta',real=True)
x=r*smp.cos(theta)+1
y=r*smp.sin(theta)
integ=smp.integrate(r,(z,0,3-r**2-2*r*smp.cos(theta)),(r,0,1),(theta,0,2*smp.pi)).evalf()
print(integ)

fig=plt.figure(figsize=(12,6))
ax=fig.add_subplot(111,projection='3d')

theta_vals=np.linspace(0,2*np.pi,100)
r_val=np.linspace(0,1,100)
R,THETA=np.meshgrid(r_val,theta_vals)
X=R*np.cos(THETA)+1
Y=R*np.sin(THETA)
Z=4-X**2-Y**2

ax.plot_surface(X,Y,Z,cmap='cool',alpha=0.7)

th=np.linspace(0,2*np.pi,100)
x2=np.cos(th)+1
y2=np.sin(th)
Z2=np.linspace(0,4,100)
for z in Z2:
    ax.plot(x2,y2,z,color='cyan',alpha=0.4) 

plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')    
plt.show()