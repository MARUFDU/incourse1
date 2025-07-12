import pulp

M = 1e6  # Big M value

# Define the problem
prob = pulp.LpProblem("Coal_Blending_Profit_Maximization_BigM", pulp.LpMaximize)

# Variables
x1 = pulp.LpVariable('Coal_A', lowBound=0)
x2 = pulp.LpVariable('Coal_B', lowBound=0)
x3 = pulp.LpVariable('Coal_C', lowBound=0)
a1 = pulp.LpVariable('Artificial_1', lowBound=0)

# Objective function with Big M penalty
prob += 12*x1 + 15*x2 + 14*x3 - M*a1

# Constraints (first as equality using artificial variable)
prob += x1 + x2 + x3 + a1 == 100, "Total_Coal_Equality"
prob += -x2 + 2*x3 <= 0, "Ash_Limit"
prob += -0.01*x1 + 0.01*x2 <= 0, "Phosphorous_Limit"

# Solve
prob.solve()

print("Status:", pulp.LpStatus[prob.status])
print(f"Coal A (x1): {x1.varValue:.2f} tons")
print(f"Coal B (x2): {x2.varValue:.2f} tons")
print(f"Coal C (x3): {x3.varValue:.2f} tons")
print(f"Artificial_1 (should be zero): {a1.varValue:.6f}")
print(f"Maximum Profit (Big-M penalized): {pulp.value(prob.objective):.2f} BDT")
