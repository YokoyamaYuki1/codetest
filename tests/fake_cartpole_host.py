import random
import sys


MAX_STEPS = 5


def emit_obs() -> None:
    values = [random.uniform(-0.1, 0.1) for _ in range(4)]
    print("obs", *values, flush=True)


def main() -> None:
    steps = 0
    for line in sys.stdin:
        line = line.strip()
        if line == "r":
            steps = 0
            emit_obs()
        elif line.startswith("s "):
            steps += 1
            if steps >= MAX_STEPS:
                print("done", flush=True)
            else:
                emit_obs()
        elif line == "q":
            break


if __name__ == "__main__":
    main()
