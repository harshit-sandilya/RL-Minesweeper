import json
import re
import time
import getpass
import os
from dotenv import load_dotenv
from minesweeper import create, move, check_over, mine_key, unknown_key
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔐 Load API key from environment or prompt
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# 🤖 LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_output_tokens=100,
    temperature=0.0,
    top_p=1.0,
    top_k=40
)

# 📊 Format board for LLM display
def format_view_for_llm(view):
    def cell_repr(cell):
        if cell == unknown_key:
            return "■"
        elif cell == mine_key:
            return "*"
        elif cell == 0:
            return "."
        else:
            return str(cell)

    header = "     " + " ".join(f"{i}" for i in range(len(view)))
    lines = [header]
    for i, row in enumerate(view):
        row_str = " ".join(cell_repr(cell) for cell in row)
        lines.append(f"{i:2} | {row_str}")
    return "\n".join(lines)

# 🔁 Ask LLM for move
def run_llm_move(view, queue):
    view_str = format_view_for_llm(view)
    prompt = f"""
You are playing Minesweeper. Here is the 8×8 grid:

{view_str}

■ = unknown cell
* = mine (avoid)
. = empty cell
1–8 = number of surrounding mines

You have already played at:
{queue}

Reply with your next move in **JSON** format:

{{ "row": x, "column": y }}

- Use zero-based indexing.
- Do not return a cell already visible or one you have already played.
- Only return raw JSON.
- Do NOT use backticks, markdown, explanations, or any extra text.
- Only valid JSON starting with {{ and ending with }}.
"""
    print("\n📤 Prompt to LLM:\n", prompt)
    response = llm.invoke(prompt)
    if not response or not response.content or not isinstance(response.content, str):
        print("❌ No response from LLM.")
        return None
    response_text = response.content.strip()

    print("\n📥 Raw Response from LLM:\n", response_text)

    # 🧹 Remove markdown backticks if present
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?", "", response_text.strip(), flags=re.IGNORECASE)
        response_text = response_text.strip("` \n")

    try:
        move = json.loads(response_text)
        print("✅ Parsed Move:", move)
        return move
    except json.JSONDecodeError as e:
        print("❌ Failed to parse JSON:", e)
        print("🔎 Raw cleaned response:", response_text)
        return None

# 🎮 Play one game and save it
def play_a_game(game_index):
    queue = set()
    index = 0
    grid, view = create(8, 10, queue, index, False)
    done = False
    game_log = []

    print("🎲 New game created with 8x8 grid and 10 mines.")
    print("🔄 Starting a new game...")

    while not done:
        print("\n🔍 Current board view:")
        print(format_view_for_llm(view))

        llm_move = run_llm_move(view, queue)
        if not llm_move:
            print("❌ Invalid move format. Skipping this turn.")
            time.sleep(2)
            continue

        if "row" not in llm_move or "column" not in llm_move:
            print("❌ Move must contain 'row' and 'column'.")
            continue

        row, col = llm_move["row"], llm_move["column"]
        if not (0 <= row < len(view) and 0 <= col < len(view[0])):
            print(f"❌ Move out of bounds: ({row}, {col})")
            continue

        # ✅ Log the step
        step = {
            "view_before_move": [[int(cell) for cell in r] for r in view],
            "move": {"row": row, "column": col}
        }

        # Execute the move
        over = move((row, col), grid, view, queue)
        step["hit_mine"] = over
        game_log.append(step)

        if over:
            print("💥 You hit a mine! Game over.")
            done = True
        elif check_over(view, 10):
            print("🎉 Congratulations! You've cleared the board!")
            done = True
        else:
            print(f"✅ Moved to ({row}, {col}). Keep going!")

        time.sleep(3)

    # ✅ Save this game's log immediately
    file_path = "llm_minesweeper_games.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            current_logs = json.load(f)
    else:
        current_logs = []

    current_logs.append(game_log)

    with open(file_path, "w") as f:
        json.dump(current_logs, f, indent=2)

    print(f"💾 Game {game_index + 1} saved to {file_path}")
    print("🔚 Game ended.")

# 🚀 Main loop
if __name__ == "__main__":
    print("🚀 Starting Minesweeper LLM Player...")

    for i in range(50):
        print(f"\n🌟 Game {i + 1}/50")
        play_a_game(i)
        time.sleep(5)

    print("🎉 All games completed!")
    print("👋 Thanks for playing!")
