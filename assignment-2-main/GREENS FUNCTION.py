import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt
x,y,theta,r=smp.symbols('x y theta r',real=True)
F1=smp.exp(x)-y**3
F2=smp.cos(y)+x**3
r_lower=0
r_upper=1
theta_lower=0
theta_upper=2*smp.pi
integrand=smp.diff(F2,x)-smp.diff(F1,y)
integrand_2=3*r**2
integrand_2=integrand_2*r*smp.diff(r)*smp.diff(theta)
integral=smp.integrate(integrand_2,(r,r_lower,r_upper),(theta,theta_lower,theta_upper)).doit().simplify()
print(f'the work done is: {integral}\n')

Theta=np.linspace(0,2*np.pi,100)
X=np.cos(Theta)
Y=np.sin(Theta)
plt.plot(X,Y,color='red')

x1=np.linspace(-1,1,20)
x2=np.linspace(-1,1,20)
X1,Y1=np.meshgrid(x1,x2)
U=np.exp(X1)-Y1**3
V=np.cos(Y1)+X1**3
plt.quiver(X1,Y1,U,V,alpha=0.5)

plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()
=====================================================
import numpy as np
import scipy as sp
import sympy as smp
import matplotlib.pyplot as plt

theta,phi=smp.symbols('theta phi',real=True)
x,y,z,G=smp.symbols('x y z G',cls=smp.Function,real=True)
x=x(phi,theta)
y=y(phi,theta)
z=z(phi,theta)
x=smp.sin(phi)*smp.cos(theta)
integrand=(x**2)*smp.sin(phi)

integrand_2=smp.lambdify([theta,phi],integrand)

surface_integral=sp.integrate.dblquad(integrand_2,0,np.pi,lambda theta:0,lambda theta:2*np.pi)[0]
print(surface_integral)

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection='3d')

th=np.linspace(0,2*np.pi,100)
ph=np.linspace(0,np.pi,100)
x_sphere=np.outer(np.cos(th),np.sin(ph))
y_sphere=np.outer(np.sin(th),np.sin(ph))
z_sphere=np.outer(np.ones(np.size(th)),np.cos(ph))

x_squared = x_sphere**2

ax.plot_surface(x_sphere, y_sphere, z_sphere, facecolors=plt.cm.plasma(x_squared), rstride=1, cstride=1, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Unit Sphere with $x^2$ Color Map')

mappable = plt.cm.ScalarMappable(cmap='plasma')
mappable.set_array(x_squared)
plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, label='$x^2$')

plt.show()
========================================
import numpy as np
import sympy as smp
import scipy as sp
import matplotlib.pyplot as plt

x,y,r,theta,z=smp.symbols('x y r theta z',real=True)
F1=x**3
F2=y**3
F3=z**2

div_F=smp.diff(F1,x)+smp.diff(F2,y)+smp.diff(F3,z)
div_F=3*r**2+2*z
print(div_F)
integrand=div_F*r
integrand_2=smp.lambdify([r,theta,z],integrand)
flux=sp.integrate.tplquad(integrand_2,0,2,0,2*np.pi,0,3)[0]
print(f'The outward flux of the vector field across the region is {flux} \n')



fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

th=np.linspace(0,2*np.pi,100)
z_vals=np.linspace(0,2,100)
x_cyl=3*np.cos(th)
y_cyl=3*np.sin(th)
for z1 in z_vals:
 ax.plot(x_cyl,y_cyl,z1,color='red',alpha=1)

x_plane=np.linspace(-3,3,100)
y_plane=np.linspace(-3,3,100)
x_plane,y_plane=np.meshgrid(x_plane,y_plane)
z_top=np.full_like(x_plane,2)
z_bottom=np.full_like(x_plane,0)

ax.plot(x_plane,y_plane,z_top,color='pink',alpha=0.5)
ax.plot(x_plane,y_plane,z_bottom,color='pink',alpha=0.5)

x11,y11,z11=0,0,2
F11=F1.subs([(x,x11),(y,y11),(z,z11)])
F21=F2.subs([(x,x11),(y,y11),(z,z11)])
F31=F3.subs([(x,x11),(y,y11),(z,z11)])

ax.quiver(x11,y11,z11,F11,F21,F31,length=0.1,color='blue')

x22,y22,z22=3,0,1
F12=F1.subs([(x,x22),(y,y22),(z,z22)])
F22=F2.subs([(x,x22),(y,y22),(z,z22)])
F32=F3.subs([(x,x22),(y,y22),(z,z22)])

ax.quiver(x22,y22,z22,F12,F22,F32,length=0.05,color='blue')

x33,y33,z33=-3,0,1
F1_func=smp.lambdify([x,y,z],F1)
F2_func=smp.lambdify([x,y,z],F2)
F3_func=smp.lambdify([x,y,z],F3)

u1=F1_func(x33,y33,z33)
v1=F2_func(x33,y33,z33)
w1=F3_func(x33,y33,z33)

ax.quiver(x33,y33,z33,u1,v1,w1,length=0.05,color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Region Enclosed by Cylinder and Planes with Vector Field Quivers')
ax.set_zlim(0,2)
plt.show()

plt.show()
======================================
import sympy as smp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy.physics.vector import ReferenceFrame
from sympy.physics.vector import curl
from mpl_toolkits.mplot3d import Axes3D

x,y,z,r,t=smp.symbols('x y z r t',real=True)
R=ReferenceFrame('R')
F=2*R[2]*R.x+3*R[0]*R.y+5*R[1]*R.z
C=curl(F,R)
print(f'curl of F is: {C}\n')

curl_F=smp.Matrix([5,2,3])

F_k=smp.Matrix([0, 6*smp.cos(t), 10*smp.sin(t)])
rr=smp.Matrix([2*smp.cos(t),2*smp.sin(t),0])
dr=rr.diff(t)
integrand_1=F_k[0]*rr.diff(t)[0]+F_k[1]*rr.diff(t)[1]+F_k[2]*rr.diff(t)[2]
line_integral=smp.integrate(integrand_1,(t,0,2*smp.pi)).doit().simplify()
print(f'The line integral is: {line_integral}\n')

z=4-x**2-y**2
normal_vec=smp.Matrix([-z.diff(x),-z.diff(y),1])
F_dot_n=curl_F.dot(normal_vec)
integrand_2=r*(F_dot_n.subs([(x,r*smp.cos(t)),(y,r*smp.sin(t))]))
surface_integral=smp.integrate(integrand_2,(r,0,2),(t,0,2*smp.pi)).doit().simplify()
print(f'The surface integral is: {surface_integral}\n')

if line_integral-surface_integral==0:
    print('Since the line and surface integrals are equal to each other,\n Stokes Theorem is verified.')
else:
    print('Stokes theorem is not verified')

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection='3d')
x=np.linspace(-2,2,100)
y=np.linspace(-2,2,100)
X,Y=np.meshgrid(x,y)
Z=4-X**2-Y**2
Z[Z<0]=np.nan
ax.plot_surface(X,Y,Z,cmap='viridis',alpha=0.5)


theta = np.linspace(0, 2*np.pi, 100)
x_circle = 2 * np.cos(theta)
y_circle = 2 * np.sin(theta)
z_circle = np.zeros_like(theta)
ax.plot(x_circle, y_circle, z_circle, color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Paraboloid and Boundary Circle')

plt.show()
