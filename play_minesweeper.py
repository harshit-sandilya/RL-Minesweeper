from minesweeper import create, move, mine_key, unknown_key

def print_board(board):
    print("\n   " + " ".join([f"{i:2}" for i in range(len(board))]))
    for i, row in enumerate(board):
        row_str = " ".join(
            ["■" if cell == unknown_key else "*" if cell == mine_key else "." if cell == 0 else str(cell) for cell in row]
        )
        print(f"{i:2} {row_str}")
    print()

def check_win(view, grid):
    for r in range(len(grid)):
        for c in range(len(grid)):
            if view[r][c] == unknown_key and grid[r][c] != mine_key:
                return False
    return True

def play_game(n=8, mines=10):
    queue = set()
    grid, view = create(n, mines)

    while True:
        print_board(view)
        try:
            r = int(input("Enter row: "))
            c = int(input("Enter column: "))
        except ValueError:
            print("❌ Invalid input. Try again.")
            continue

        if not (0 <= r < n and 0 <= c < n):
            print("❌ Out of bounds.")
            continue

        game_over = move((r, c), grid, view, queue)

        if game_over:
            print_board(grid)
            print("💥 Game Over! You hit a mine.")
            break

        if check_win(view, grid):
            print_board(view)
            print("🎉 You Win! All safe cells revealed.")
            break

if __name__ == "__main__":
    play_game()
