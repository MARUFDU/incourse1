import numpy as np

# Data
cost = np.array([
    [4, 3, 1, 2, 6],
    [5, 2, 3, 4, 5],
    [3, 5, 6, 3, 2],
    [2, 4, 4, 5, 3]
])
supply = [80, 60, 40, 20]
demand = [60, 60, 30, 40, 10]

def north_west_corner(cost, supply, demand):
    m, n = cost.shape
    allocation = np.zeros((m, n), dtype=int)
    i = j = 0
    sup = supply.copy()
    dem = demand.copy()
    while i < m and j < n:
        x = min(sup[i], dem[j])
        allocation[i, j] = x
        sup[i] -= x
        dem[j] -= x
        if sup[i] == 0 and i < m-1:
            i += 1
        elif dem[j] == 0 and j < n-1:
            j += 1
        else:
            break
    return allocation

def least_cost_method(cost, supply, demand):
    m, n = cost.shape
    allocation = np.zeros((m, n), dtype=int)
    sup = supply.copy()
    dem = demand.copy()
    cost_cp = cost.copy()
    while np.sum(allocation) < sum(supply):
        min_cost = np.inf
        for i in range(m):
            for j in range(n):
                if sup[i] > 0 and dem[j] > 0 and cost_cp[i, j] < min_cost:
                    min_cost = cost_cp[i, j]
                    min_cell = (i, j)
        i, j = min_cell
        x = min(sup[i], dem[j])
        allocation[i, j] = x
        sup[i] -= x
        dem[j] -= x
        if sup[i] == 0:
            cost_cp[i, :] = np.inf  # Remove exhausted row
        if dem[j] == 0:
            cost_cp[:, j] = np.inf  # Remove exhausted column
    return allocation

def print_solution(allocation, cost, method_name):
    total_cost = np.sum(allocation * cost)
    print(f"\nSolution by {method_name}:")
    print("Allocation Matrix (rows: A-D, columns: P-T):")
    print(allocation)
    print(f"Total Transportation Cost: {total_cost}")

# Run North-West Corner
nw_allocation = north_west_corner(cost, supply, demand)
print_solution(nw_allocation, cost, "North-West Corner Rule")

# Run Least Cost Method
lc_allocation = least_cost_method(cost, supply, demand)
print_solution(lc_allocation, cost, "Least Cost Method")
