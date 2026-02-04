import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_zoo3.train import train

if __name__ == "__main__":
    train()
