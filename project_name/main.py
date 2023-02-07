from tqdm import tqdm


def run(n):

    if n >= 5:
        raise ValueError("n should be smaller than 5")

    for i in tqdm(range(n)):
        print(i)

    return n


if __name__ == "__main__":

    run()
