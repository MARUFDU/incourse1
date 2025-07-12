import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)

f1 = 4 * X**2 + Y**2

z_levels=[1,4,9,16,25,36]
fig=plt.figure(figsize=(12,6))
ax=fig.add_subplot(121)
contour1 = plt.contour(X, Y, f1, levels=z_levels, cmap='viridis')
plt.clabel(contour1, inline=True, fontsize=8)
plt.title('Contour plot of f(x, y) = 4x^2 + y^2')
plt.xlabel('x')
plt.ylabel('y')

def level_surface(x,y,k):
    return np.sqrt(x**2+y**2+k)


alpha_values=np.linspace(0.3,0.8,len(z_levels))
ax=fig.add_subplot(122,projection='3d')

for k,alpha in zip(z_levels,alpha_values):
    Z=level_surface(X,Y,k)
    ax.plot_surface(X,Y,Z,alpha=alpha,cmap='coolwarm',edgecolor='none')
    ax.plot_surface(X,Y,-Z,alpha=alpha,cmap='coolwarm',edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Level surfaces of f(x,y,z)=z^2-x^2-y^2')

plt.tight_layout()
plt.show()

=============================================================
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

# Define the range for x and y for function (a)
x_a = np.linspace(1, 7, 400)
y_a = np.linspace(1, 7, 400)
X_a, Y_a = np.meshgrid(x_a, y_a)

# Define the function f(x, y) = y^2 - 2y * cos(x)
Z_a = Y_a**2 - 2 * Y_a * np.cos(X_a)

# Define the range for x and y for function (b)
x_b = np.linspace(0, 2 * np.pi, 400)
y_b = np.linspace(0, 2 * np.pi, 400)
X_b, Y_b = np.meshgrid(x_b, y_b)

# Define the function g(x, y) = |sin(x) * sin(y)|
Z_b = np.abs(np.sin(X_b) * np.sin(Y_b))

# Create the 3D plot for function (a)
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_a, Y_a, Z_a, cmap='viridis')
ax1.set_title('3D plot of f(x, y) = y^2 - 2y * cos(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x, y)')

# Create the 3D plot for function (b)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_b, Y_b, Z_b, cmap='viridis')
ax2.set_title('3D plot of g(x, y) = |sin(x) * sin(y)|')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('g(x, y)')

plt.tight_layout()
plt.show()
======================================================

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
x,y=sp.symbols('x y',real=True)

f1=4*x*y-x**4-y**4
f1_x=sp.diff(f1,x)
f1_y=sp.diff(f1,y)
f1_xx=sp.diff(f1_x,x)
f1_xy=sp.diff(f1_x,y)
f1_yy=sp.diff(f1_y,y)

crit_points_1=sp.solve([f1_x,f1_y],(x,y))

f2 = 4*x**2*sp.exp(y) - 2*x**4 - sp.exp(4*y)
f2_x = sp.diff(f2, x)
f2_y = sp.diff(f2, y)
f2_xx = sp.diff(f2_x, x)
f2_yy = sp.diff(f2_y, y)
f2_xy = sp.diff(f2_x, y)

crit_points_2 = sp.solve([f2_x, f2_y], (x, y))

def classification(fxx,fyy,fxy,cp):
    H=fxx*fyy-fxy**2
    for point in cp:
        c1=H.subs({x: point[0],y: point[1]})
        if c1>0 and fxx.subs({x: point[0],y: point[1]})>0:
            print(f'relative minima at {point}')
        elif c1>0 and fxx.subs({x: point[0],y: point[1]})<0:
            print(f'relative maxima at {point}')
        elif c1<0:
            print(f'saddle point at {point}')
        elif c1==0:
           print('no conclusion')
        else:
           print('invalid')

print('for f1: ')
classification(f1_xx,f1_yy,f1_xy,crit_points_1) 
print('\n for f2: ')
classification(f2_xx,f2_xy,f2_yy,crit_points_2)           

x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z1 = 4*X*Y - X**4 - Y**4
Z2 = 4*X**2*np.exp(Y) - 2*X**4 - np.exp(4*Y)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
ax1.set_title("Surface plot of f(x,y) = 4xy - x^4 - y^4")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x, y)")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
ax2.set_title("Surface plot of f(x,y) = 4x^2 e^y - 2x^4 - e^{4y}")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x, y)")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax1 = axes[0]
ax1.contour(X, Y, Z1, levels=30, cmap="viridis")


for point in crit_points_1:
 ax1.plot(point[0], point[1], 'ro')

ax1.set_title("Contour plot of f(x,y) = 4xy - x^4 - y^4")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

ax2 = axes[1]
ax2.contour(X, Y, Z2, levels=30, cmap="plasma")

for point in crit_points_2:
 ax2.plot(point[0], point[1], 'ro')

ax2.set_title("Contour plot of f(x,y) = 4x^2 e^y - 2x^4 - e^{4y}")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.tight_layout()
plt.show()

