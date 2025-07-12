import numpy as np
import sympy as smp
import math

#a
t=smp.symbols('t',real=True)
pie=round(math.pi,2)
r1=smp.Matrix([smp.log(t),smp.exp(-t),t**3])
v1=smp.diff(r1,t)
r2=smp.Matrix([2*smp.cos(pie*t),2*smp.sin(pie*t),3*t])
v2=smp.diff(r2,t)

t1_0=2
t2_0=1/3

def tangent_line(r,v,t0):
    r0=r.subs(t,t0)
    v0=v.subs(t,t0)
    return r0+v0*t

t_line=tangent_line(r1,v1,t1_0)
print(f"The parametric equation of the tangent line to the curve at t0 = {t1_0} is:")
print(f"x(t) = {t_line[0]}")
print(f"y(t) = {t_line[1]}")
print(f"z(t) = {t_line[2]}")

t_line2=tangent_line(r2,v2,t2_0)
print(f"The parametric equation of the tangent line to the curve at t0 = {t2_0} is:")
print(f"x(t) = {t_line2[0]}")
print(f"y(t) = {t_line2[1]}")
print(f"z(t) = {t_line2[2]}")

#b
n1=np.array([3,-6,-2])
n2=np.array([2,1,-2])

parallel_vector=np.cross(n1,n2)

print(f"\n the vector parallel to the line of intersection of the planes is {parallel_vector[0]}i+{parallel_vector[1]}j+{parallel_vector[2]}k \n")


#c
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

t=sp.symbols('t',real=True)

r=-sp.Matrix([3*t,sp.sin(t),t**2])
v=r.diff(t)
a=v.diff(t)

print(f"position vector r(t)={r}\n")
print(f"velocity vector v(t)={v}\n")
print(f"acceleration vector a(t)={a}\n")

def dot_prod(a,b):
    return a.dot(b)

def magnitude(a):
    return sp.sqrt(a.dot(a))

def theta(t_val):
    v_t=v.subs(t,t_val)
    a_t=a.subs(t,t_val)
    cos_theta=dot_prod(v_t,a_t)/(magnitude(v_t)*magnitude(a_t))
    return sp.acos(cos_theta)

t_values=np.linspace(0.1,10,400)
theta_values=[]

for t_val in t_values:
    theta_val=theta(t_val).evalf()
    theta_values.append(theta_val)


plt.plot(t_values,theta_values,'--',c='red',label=r'$\theta(t)$')
plt.xlabel('t')
plt.ylabel(r'$\theta(t)$')
plt.legend()
plt.show()

