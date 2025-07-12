import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

t = sp.symbols('t',real=True)

r=sp.Matrix([sp.exp(t), sp.exp(t)*sp.cos(t), sp.exp(t)*sp.sin(t)])
print(r)
r_prime=r.diff(t)
r_prime2=r.diff(t,2)
print(r_prime2)
r_prime3=r.diff(t,3)
print(r_prime3)

def Tangent(r,tval):
 T=r/(r.norm())
 return T.subs(t,tval).evalf()

def Normal(r,tval):
 T=r/(r.norm())
 N=T.diff(t)
 N=N/N.norm()
 return N.subs(t,tval).evalf()

def Binormal(r,tval):
 T=r/(r.norm())
 N=T.diff(t)
 N=N/N.norm()
 B=T.cross(N)
 return B.subs(t,tval).evalf()

def curvature(r1,r2,tval):
 kappa=(r1.cross(r2)).norm()/(r1.norm()**3)
 return kappa.subs(t,tval).evalf()

def torsion(r1,r2,r3,tval):
 tao= r1.dot(r2.cross(r3))/(r1.cross(r2)).norm()**2
 return tao.subs(t,tval).evalf()

print(f'unit tangent vector at t=0 is : {Tangent(r_prime,0)} \n')
print(f'unit normal vector at t=0 is : {Normal(r_prime,0)} \n')
print(f'binormal vector at t=0 is : {Binormal(r_prime,0)} \n')
print(f'curvature at t=0 is : {curvature(r_prime,r_prime2,0)} \n')
print(f'torsion at t=0 is : {torsion(r_prime,r_prime2,r_prime3,0)} \n')

w=sp.Matrix([2*sp.cos(t),3*sp.sin(t),0])
w_prime=sp.diff(w,t)
w_prime2=sp.diff(w,t,2)
w_prime3=sp.diff(w,t,3)
print(f'for w: \n')

# At t=0
print(f'unit tangent vector at t=0 is : {Tangent(w_prime,0)} \n')
print(f'unit normal vector at t=0 is : {Normal(w_prime,0)} \n')
print(f'binormal vector at t=0 is : {Binormal(w_prime,0)} \n')
print(f'curvature at t=0 is : {curvature(w_prime,w_prime2,0)} \n')
print(f'torsion at t=0 is : {torsion(w_prime,w_prime2,w_prime3,0)} \n')

#t=2*pi
print(f'unit tangent vector at t=2*pi is : {Tangent(w_prime,2*sp.pi)} \n')
print(f'unit normal vector at t=2*pi is : {Normal(w_prime,2*sp.pi)} \n')
print(f'binormal vector at t=2*pi is : {Binormal(w_prime,2*sp.pi)} \n')
print(f'curvature at t=2*pi is : {curvature(w_prime,w_prime2,2*sp.pi)} \n')
print(f'torsion at t=2*pi is : {torsion(w_prime,w_prime2,w_prime3,2*sp.pi)} \n')

t_values = np.linspace(0, 2 * np.pi, 400)

kappa1_values=[]
for t_val in t_values:
 kappa1_values.append(curvature(r_prime,r_prime2,t_val)) 

kappa2_values =[]
for t_val in t_values:
 kappa2_values.append(curvature(w_prime,w_prime2,t_val)) 

plt.figure(figsize=(10, 6))
plt.plot(t_values, kappa1_values, label='Curvature of r1(t)')
plt.plot(t_values, kappa2_values, label='Curvature of r2(t)')
plt.xlabel('t')
plt.ylabel('Curvature κ(t)')
plt.title('Curvature of the Curves')
plt.legend()
plt.grid(True)
plt.show()
=====================================================
import sympy as sp
x, y = sp.symbols('x y',real=True)
f = y**2 * sp.cos(x - y)
f_x = f.diff(x)
f_y = f.diff(y)
f_xx = f_x.diff(x)
f_yy = f_y.diff(y)
f_xy = f_x.diff(y)
f_yx = f_y.diff(x)
laplace_eq = f_xx + f_yy

if laplace_eq==0:
    print('laplace eqn satisfied')
else:
    print('laplace eqn not satisfied')

# Check Cauchy-Riemann equations
u = sp.re(f)
v = sp.im(f)

if u.diff(x)==v.diff(y) and u.diff(y)==-v.diff(x):
    print('Cauchy-Riemann conditions  satisfied')
else:
    print('Cauchy Riemann conditons not satisfied')

if f_xy==f_yx:
    print('fxy=fyx')
else:
    print('fxy != fyx')
=========================================================

import sympy as sp

t=sp.symbols('t',real=True)

x=sp.cos(t)
y=sp.sin(t)
z=sp.tan(t)

w=sp.sqrt((x**2)+(y**2)+(z**2))

w_diff=w.diff(t)

print(w_diff)
t0=sp.pi/4
print(w_diff.subs(t,t0).evalf())












