import subprocess
from typing import List, Tuple


class CartPoleEnv:
    def __init__(self, command: str, max_steps: int = 500) -> None:
        self._process = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._max_steps = max_steps
        self._step_count = 0

    def obs_dim(self) -> int:
        return 4

    def reset(self) -> List[float]:
        self._step_count = 0
        self._send("r")
        line = self._read_line()
        return self._parse_obs(line)

    def step(self, action: int) -> Tuple[List[float] | None, bool, float]:
        self._send(f"s {action}")
        line = self._read_line()
        self._step_count += 1
        done = False
        obs: List[float] | None
        if line.strip() == "done":
            done = True
            obs = None
        else:
            obs = self._parse_obs(line)
        if self._step_count >= self._max_steps:
            done = True
        return obs, done, 1.0

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._send("q")
            except BrokenPipeError:
                pass
            self._process.terminate()
            self._process.wait(timeout=2)

    def _send(self, message: str) -> None:
        if self._process.stdin is None:
            raise RuntimeError("Process stdin is closed")
        self._process.stdin.write(message + "\n")
        self._process.stdin.flush()

    def _read_line(self) -> str:
        if self._process.stdout is None:
            raise RuntimeError("Process stdout is closed")
        line = self._process.stdout.readline()
        if line == "":
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read()
            raise RuntimeError(f"Process terminated unexpectedly: {stderr}")
        return line

    @staticmethod
    def _parse_obs(line: str) -> List[float]:
        parts = line.strip().split()
        if not parts or parts[0] != "obs" or len(parts) != 5:
            raise ValueError(f"Unexpected observation line: {line}")
        return [float(value) for value in parts[1:]]

    def __del__(self) -> None:
        self.close()
