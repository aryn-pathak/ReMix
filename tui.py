import curses
import math
import string

ANGLE_MIN, ANGLE_MAX = 45, 315  # azimuth runs the long way round, through the back
ANGLE_STEP = 5   # degrees per keypress
STEP = 0.1       # meters per keypress, for distance and height
MIN_DISTANCE = 1.0
WALL_GAP = 0.1   # keep sources this far off the floor and ceiling
SLIDER = 30      # slider width in characters
ENTER = (curses.KEY_ENTER, 10, 13)

# offsets to try when a letter's cell is taken, nearest first. A cell is about twice as
# tall as it is wide, so a row away counts double and letters end up side by side
NEARBY = sorted(((dr, dc) for dr in range(-3, 4) for dc in range(-6, 7)),
                key=lambda o: (2 * o[0]) ** 2 + o[1] ** 2)


def put(win, y, x, text, attr=0):
    # curses raises on writes past the edge (and on the bottom-right cell), clip instead
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def slider(value, lo, hi):
    pos = round((value - lo) / (hi - lo) * (SLIDER - 1)) if hi > lo else 0
    return "[" + "=" * pos + "|" + "-" * (SLIDER - 1 - pos) + "]"


def pressed_letter(key, letters):
    ch = chr(key).upper() if 0 <= key < 256 else ""
    return letters.index(ch) if ch and ch in letters else None


def choose(stdscr, prompt, options):
    curses.curs_set(0)
    i = 0
    while True:
        stdscr.erase()
        put(stdscr, 0, 0, prompt)
        x = len(prompt) + 1
        for j, option in enumerate(options):
            put(stdscr, 0, x, f" {option} ", curses.A_REVERSE if j == i else 0)
            x += len(option) + 3
        put(stdscr, 1, 0, "simple: every instrument at one distance, at its default angle", curses.A_DIM)
        put(stdscr, 2, 0, "advanced: place each instrument on a map, then set its height", curses.A_DIM)
        put(stdscr, 4, 0, "←/→ choose · Enter confirm", curses.A_DIM)

        key = stdscr.getch()
        if key == curses.KEY_LEFT:
            i = max(i - 1, 0)
        elif key == curses.KEY_RIGHT:
            i = min(i + 1, len(options) - 1)
        elif key in ENTER:
            return options[i]


def place(stdscr, room, names, azimuths):
    # returns {name: [azimuth, distance, height]}, height left at ear level
    curses.curs_set(0)
    length, width = room["length"], room["width"]
    max_d = min(length, width) / 2
    d0 = max(max_d - 1, MIN_DISTANCE)
    pos = {name: [azimuths[name], d0, 0.0] for name in names}
    letters = string.ascii_uppercase[:len(names)]
    sel = 0

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        # the room is drawn facing up: +x (length) runs up the screen, +y (width) to the left
        scale = min((w - 2) / width, 2 * (h - 10) / length)  # columns per meter
        cols, rows = int(width * scale), int(length * scale / 2)

        if cols < 3 or rows < 3:
            put(stdscr, 0, 0, "Terminal too small, enlarge the window")
        else:
            def cell(x, y):
                return (round((length - x) / length * (rows - 1)),
                        round((width - y) / width * (cols - 1)))

            put(stdscr, 0, 0, f"Room {length:g} x {width:g} m, listener facing up")
            x = 0
            for i, name in enumerate(names):
                put(stdscr, 1, x, f"{letters[i]} {name}", curses.A_REVERSE if i == sel else 0)
                x += len(name) + 4

            top = 2
            put(stdscr, top, 0, "+" + "-" * cols + "+")
            for r in range(rows):
                put(stdscr, top + 1 + r, 0, "|" + " " * cols + "|")
            put(stdscr, top + rows + 1, 0, "+" + "-" * cols + "+")

            # the selected letter keeps its true cell, the rest move aside if they collide
            taken = {cell(length / 2, width / 2): "L"}
            for i in [sel] + [i for i in range(len(names)) if i != sel]:
                az, d, _ = pos[names[i]]
                r, c = cell(length / 2 + d * math.cos(math.radians(az)),
                            width / 2 + d * math.sin(math.radians(az)))
                r, c = next(((r + dr, c + dc) for dr, dc in NEARBY
                             if 0 <= r + dr < rows and 0 <= c + dc < cols
                             and (r + dr, c + dc) not in taken), (r, c))
                taken[(r, c)] = letters[i]
            for (r, c), ch in taken.items():
                put(stdscr, top + 1 + r, 1 + c, ch,
                    curses.A_REVERSE if ch == letters[sel] else curses.A_BOLD)

            az, d, _ = pos[names[sel]]
            b = top + rows + 2
            put(stdscr, b + 1, 0, f"{letters[sel]}  {names[sel]}", curses.A_BOLD)
            put(stdscr, b + 2, 0, f"angle     {slider(az, ANGLE_MIN, ANGLE_MAX)} {az:5.0f}°")
            put(stdscr, b + 3, 0, f"distance  {slider(d, MIN_DISTANCE, max_d)} {d:5.1f} m")
            put(stdscr, b + 5, 0, f"A-{letters[-1]} pick instrument · ←/→ angle · ↑/↓ distance"
                                  " · Enter next: heights", curses.A_DIM)

        key = stdscr.getch()
        p = pos[names[sel]]
        if key == curses.KEY_LEFT:
            p[0] = max(p[0] - ANGLE_STEP, ANGLE_MIN)
        elif key == curses.KEY_RIGHT:
            p[0] = min(p[0] + ANGLE_STEP, ANGLE_MAX)
        elif key == curses.KEY_UP:
            p[1] = min(round(p[1] + STEP, 1), max_d)
        elif key == curses.KEY_DOWN:
            p[1] = max(round(p[1] - STEP, 1), MIN_DISTANCE)
        elif key in ENTER:
            return pos
        elif (i := pressed_letter(key, letters)) is not None:
            sel = i


def elevate(stdscr, room, pos):
    # edits the heights in pos ({name: [azimuth, distance, height]}) in place
    curses.curs_set(0)
    names = list(pos)
    letters = string.ascii_uppercase[:len(names)]
    order = sorted(range(len(names)), key=lambda i: pos[names[i]][0])  # anticlockwise
    top_h = room["height"] / 2 - WALL_GAP
    sel = order[0]

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        n = min(h - 7, 31)  # rows per track
        n -= 1 - n % 2      # odd, so ear level gets a row of its own
        left, gap = 8, 6
        span = gap * len(names)

        if n < 3 or w < left + span:
            put(stdscr, 0, 0, "Terminal too small, enlarge the window")
        else:
            def row(v):
                return 2 + round((top_h - v) / (2 * top_h) * (n - 1))

            put(stdscr, 0, 0, f"Height above the listener's ears ({room['height'] / 2:g} m up)")
            put(stdscr, row(top_h), 0, f"{top_h:+.1f} m")
            put(stdscr, row(-top_h), 0, f"{-top_h:+.1f} m")
            for k in range(len(names)):
                for r in range(n):
                    put(stdscr, 2 + r, left + gap * k + 2, "|", curses.A_DIM)
            put(stdscr, row(0), 0, "ears")
            put(stdscr, row(0), left, "-" * span, curses.A_DIM)
            for k, i in enumerate(order):
                c = left + gap * k + 2
                put(stdscr, row(pos[names[i]][2]), c, letters[i],
                    curses.A_REVERSE if i == sel else curses.A_BOLD)
                put(stdscr, 2 + n, c, letters[i])

            az, d, height = pos[names[sel]]
            el = math.degrees(math.atan2(height, d))
            put(stdscr, n + 4, 0, f"{letters[sel]}  {names[sel]}", curses.A_BOLD)
            put(stdscr, n + 4, 4 + len(names[sel]),
                f"height {height:+.1f} m   elevation {el:+.1f}°   ({d:g} m away at {az:g}°)")
            put(stdscr, n + 6, 0, f"A-{letters[-1]} pick instrument · ↑/↓ height · Enter start processing",
                curses.A_DIM)

        key = stdscr.getch()
        p = pos[names[sel]]
        if key == curses.KEY_UP:
            p[2] = min(round(p[2] + STEP, 1), top_h)
        elif key == curses.KEY_DOWN:
            p[2] = max(round(p[2] - STEP, 1), -top_h)
        elif key in ENTER:
            return pos
        elif (i := pressed_letter(key, letters)) is not None:
            sel = i
