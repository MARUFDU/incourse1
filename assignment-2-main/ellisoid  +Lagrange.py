import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the symbolic variables
x, y, z = sp.symbols('x y z',real=True)

# Define the ellipsoid function
F = x**2 + 4*y**2 + z**2 - 18

# Compute the gradient of F
grad_F = sp.Matrix([F.diff(x), F.diff(y), F.diff(z)])

# Evaluate the gradient at the point (1, 2, 1)
point = (1, 2, 1)
grad_F_at_point = grad_F.subs({x: point[0], y: point[1], z: point[2]})

# (a) Equation of the tangent plane
# The equation of the tangent plane is given by:
# grad_F_at_point . (x - x0, y - y0, z - z0) = 0
tangent_plane_eq = grad_F_at_point[0] * (x - point[0]) + grad_F_at_point[1] * (y - point[1]) + grad_F_at_point[2] * (z - point[2])
tangent_plane_eq = sp.simplify(tangent_plane_eq)

# (b) Parametric equations of the normal line
# The parametric equations of the normal line are given by:
# x = x0 + t * grad_F_at_point[0]
# y = y0 + t * grad_F_at_point[1]
# z = z0 + t * grad_F_at_point[2]
t = sp.symbols('t',real=True)
normal_line_eqs = [point[0] + t * grad_F_at_point[0], point[1] + t * grad_F_at_point[1], point[2] + t * grad_F_at_point[2]]

# (c) Acute angle with the xy-plane
# The angle between the normal vector and the z-axis is given by:
# cos(theta) = |grad_F_at_point[2]| / ||grad_F_at_point||
# theta = arccos(|grad_F_at_point[2]| / ||grad_F_at_point||)
grad_F_norm = sp.sqrt(grad_F_at_point.dot(grad_F_at_point))
cos_theta = abs(grad_F_at_point[2]) / grad_F_norm
theta = sp.acos(cos_theta)
theta_deg = sp.deg(theta)

# Print the results
print(f"Equation of the tangent plane: {tangent_plane_eq} = 0")
print(f"Parametric equations of the normal line: x = {normal_line_eqs[0]}, y = {normal_line_eqs[1]}, z = {normal_line_eqs[2]}")
print(f"Acute angle with the xy-plane: {theta_deg.evalf()} degrees")

# (d) Visualization
# Define the ellipsoid


# Define the tangent plane
xx, yy = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
zz = (grad_F_at_point[0] * (xx - point[0]) + grad_F_at_point[1] * (yy - point[1]) + grad_F_at_point[2] * point[2]) / -grad_F_at_point[2] + point[2]

# Define the normal line
t_vals = np.linspace(-3, 3, 100)
x_normal = point[0] + t_vals * grad_F_at_point[0]
y_normal = point[1] + t_vals * grad_F_at_point[1]
z_normal = point[2] + t_vals * grad_F_at_point[2]

# Plot the ellipsoid, tangent plane, and normal line
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
X = np.outer(np.cos(u), np.sin(v)) * np.sqrt(18) 
Y = np.outer(np.sin(u), np.sin(v)) * np.sqrt(18/4) 
Z = np.outer(np.ones(np.size(u)), np.cos(v)) * np.sqrt(18) 
ax.plot_surface(X, Y, Z, color='c', alpha=0.3)

# Plot the ellipsoid
#ax.plot_surface(12*x_ellipsoid, 12*y_ellipsoid, 12*z_ellipsoid, cmap='viridis', alpha=0.5)

# Plot the tangent plane
ax.plot_surface(xx, yy, zz, color='black', alpha=0.5)

# Plot the normal line
#ax.plot(x_normal, y_normal, z_normal, color='g', linewidth=2)

# Plot the point (1, 2, 1)
ax.scatter(point[0], point[1], point[2], label='1 ,2 ,1',color='k', s=50)

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ellipsoid, Tangent Plane, and Normal Line')
ax.legend()

plt.show()
==================================================
import sympy as sp

# Define the symbolic variables
x, y, z, λ = sp.symbols('x y z λ')

# Define the temperature function T(x, y, z)
T = 8 * x**2 + 4 * y * z - 16 * z + 600

# Define the constraint function g(x, y, z) representing the ellipsoid
g = 4 * x**2 + y**2 + 4 * z**2 - 16

# Define the Lagrange function L(x, y, z, λ)
L = T + λ * g

# Compute the partial derivatives of L
L_x = sp.diff(L, x)
L_y = sp.diff(L, y)
L_z = sp.diff(L, z)
L_λ = sp.diff(L, λ)

# Solve the system of equations L_x = 0, L_y = 0, L_z = 0, L_λ = 0
solutions = sp.solve([L_x, L_y, L_z, L_λ], (x, y, z, λ))

# Find the hottest point by evaluating T at the solutions
hottest_point = None
max_temperature = -sp.oo

for sol in solutions:
    temp = T.subs({x: sol[0], y: sol[1], z: sol[2]})
    if temp > max_temperature:
        max_temperature = temp
        hottest_point = sol

# Print the results
print(f"Hottest point: {hottest_point}")
print(f"Maximum temperature: {max_temperature}")