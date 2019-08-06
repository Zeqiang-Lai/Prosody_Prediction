if __name__ == '__main__':
    file_path = 'data/biaobei3/train/labels.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        print("Lines: {0}".format(len(lines)))

        total = " ".join(lines)
        items = total.split()

        B = items.count('B')
        I = items.count('I')
        print("B: {0}".format(B))
        print("I: {0}".format(I))
        print(I/(B+I))
        print((I+len(lines)) / (B+I))