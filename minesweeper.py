import random

mine_key = 16
unknown_key = 255

def set_mines(n, mines_no, numbers):
    count = 0
    while count < mines_no:
        val = random.randint(0, n * n - 1)
        r = val // n
        col = val % n
        if numbers[r][col] != mine_key:
            count += 1
            numbers[r][col] = mine_key

def set_values(n, numbers):
    for r in range(n):
        for col in range(n):
            if numbers[r][col] == mine_key:
                continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, col + dc
                if 0 <= nr < n and 0 <= nc < n and numbers[nr][nc] == mine_key:
                    numbers[r][col] += 1

def create(n_grid, n_mines):
    grid = [[0] * n_grid for _ in range(n_grid)]
    view = [[unknown_key] * n_grid for _ in range(n_grid)]
    set_mines(n_grid, n_mines, grid)
    set_values(n_grid, grid)
    return grid, view

def neighbours(r, col, queue, grid, view, n):
    if (r, col) not in queue:
        queue.add((r, col))
        if grid[r][col] == 0:
            view[r][col] = grid[r][col]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, col + dc
                if 0 <= nr < n and 0 <= nc < n:
                    neighbours(nr, nc, queue, grid, view, n)
        else:
            view[r][col] = grid[r][col]

def move(action, grid, view, queue):
    row, col = action
    view[row][col] = grid[row][col]
    if grid[row][col] == mine_key:
        queue.add((row, col))
        return True
    elif grid[row][col] == 0:
        neighbours(row, col, queue, grid, view, len(grid))
        return False
    else:
        queue.add((row, col))
        return False

def check_over(mine_values, mines_no):
    count = sum(cell == unknown_key for row in mine_values for cell in row)
    return count == mines_no
