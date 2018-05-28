counts = {'1': 0, '2': 0, '3': 0, '4': 0}

with open('data/labels_done.txt', 'r') as f:
    for line in f:
        line = line.split()[1]
        for label in line:
            counts[label] += 1

for key in counts:
    print('Label: {}, Probability: {:.2f} %'.format(key, counts[key]/1750))
