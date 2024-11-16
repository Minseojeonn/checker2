import os
from fire import Fire


def main(
):

    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   
    eps = [1e-1, 1e-3, 1e-5, 0.2]
    seeds = [10, 20, 30]
    for a in alpha:
        for e in eps:
            os.system(f"python main.py --alpha {a} --eps {e}")

if __name__ == "__main__":
    Fire(main)