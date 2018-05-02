import numpy as np

def interpolate(p1, p2):
    m = (p2 - p1) / 10
    interp = [p1 + i * m for i in range(10)]
    return np.array(interp)

def fill_missing(labels):
    labels = [int(c) for c in labels]
    labels_interp = np.zeros(len(labels)*10)
    labels_interp[:10] = labels[0]
    for i in range(len(labels)-1):
        labels_interp[(i+1)*10: (i+2)*10] = interpolate(labels[i], labels[i+1])

    labels_interp = np.round(labels_interp+0.01)
    labels_interp = labels_interp.astype(int)
    labels_interp = [str(c) for c in labels_interp]
    labels_interp = ''.join(labels_interp)
    return labels_interp


lines = []
with open('data/labels_done.txt', 'r') as f:
    for line in f:
        line = line.split()
        labels = line[1]
        labels_interp = fill_missing(labels)
        new_line = line[0] + ' ' + labels_interp + '\n'
        lines.append(new_line)


with open('data/labels_new.txt', 'w') as f:
    for line in lines:
        f.write(line)
