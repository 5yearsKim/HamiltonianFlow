import numpy as np
import matplotlib.pyplot as plt


def moon1(num_samples):
    x1 = np.random.normal(0, 3, num_samples // 2)
    x2 = np.zeros(num_samples // 2)
    for idx in range(num_samples // 2):
        x2[idx] = np.random.normal(.2 * x1[idx] ** 2, 1)
    x1 -= 0
    x2 -= 40

    y1 = np.random.normal(0, 3, num_samples // 2)
    y2 = np.zeros(num_samples // 2)
    for idx in range(num_samples // 2):
        y2[idx] = np.random.normal(-.2 * y1[idx] ** 2, 1)
    y1 += 0
    y2 += 40

    merged = np.array([np.append(x1, y1), np.append(x2, y2)]).T
    merged = (merged - np.mean(merged, axis=0)) / np.std(merged, axis=0)
    merged[:, 0] /= 3.5
    merged *= 3
    return merged


def moon2(num_samples):
    x1 = np.random.normal(0, 3, num_samples // 2)
    x2 = np.zeros(num_samples // 2)
    for idx in range(num_samples // 2):
        x2[idx] = np.random.normal(.3 * x1[idx] ** 2, 1)
    x1 -= 10
    x2 -= 40

    y1 = np.random.normal(0, 3, num_samples // 2)
    y2 = np.zeros(num_samples // 2)
    for idx in range(num_samples // 2):
        y2[idx] = np.random.normal(-.3 * y1[idx] ** 2, 1)
    y1 += 10
    y2 += 40

    merged = np.array([np.append(x1, y1), np.append(x2, y2)]).T
    merged = (merged - np.mean(merged, axis=0)) / np.std(merged, axis=0) * 2
    return merged

def stars(num_samples):
    num_groups = 6
    size = 4
    x = np.array([])
    y = np.array([])
    for i in range(num_groups):
        x = np.append(x, np.random.normal(size * np.cos(2*np.pi/num_groups * i), 0.3, num_samples // num_groups ))
        y = np.append(y, np.random.normal(size * np.sin(2*np.pi/num_groups * i), 0.3, num_samples // num_groups ))
    merged = np.array([x + 16.,y]).T
#	merged =  (merged - np.mean(merged, axis=0)) / np.std(merged, axis=0) * 2
    return merged

if __name__ == "__main__":
    dataset = stars(num_samples=102)
    plt.scatter(dataset[:, 0], dataset[:, 1], s= 10)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    # plt.savefig("moon2_ref")
    plt.show()

    # np.save('moon1.npy', merged)
