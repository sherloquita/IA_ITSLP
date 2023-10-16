from collections import deque
import copy

# Representación del estado inicial y objetivo
initial_state = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
goal_state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# Función para verificar si dos estados son iguales
def states_equal(state1, state2):
    return all(state1[i][j] == state2[i][j] for i in range(3) for j in range(3))

# Función para obtener los movimientos posibles desde un estado dado
def get_possible_moves(state):
    zero_pos = None
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                zero_pos = (i, j)
                break
    moves = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for d in directions:
        new_i, new_j = zero_pos[0] + d[0], zero_pos[1] + d[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_state = copy.deepcopy(state)
            new_state[zero_pos[0]][zero_pos[1]] = state[new_i][new_j]
            new_state[new_i][new_j] = 0
            moves.append(new_state)

    return moves

# Función para resolver el rompecabezas utilizando BFS
def solve_puzzle(initial_state, goal_state):
    queue = deque([(initial_state, [])])
    visited = set()

    while queue:
        current_state, path = queue.popleft()

        if states_equal(current_state, goal_state):
            return path

        visited.add(tuple(map(tuple, current_state)))

        for next_state in get_possible_moves(current_state):
            if tuple(map(tuple, next_state)) not in visited:
                queue.append((next_state, path + [next_state]))

# Resuelve el rompecabezas e imprime la solución
solution_path = solve_puzzle(initial_state, goal_state)
for i, state in enumerate(solution_path):
    print(f"Paso {i + 1}:\n")
    for row in state:
        print(row)
    print()