import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_adjacency_matrix(points):
    size = len(points)
    adjacency_matrix = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i != j:
                distance = calculate_distance(points[i], points[j])
                adjacency_matrix[i][j] = distance
    
    return adjacency_matrix


def main():
    with open("hello_400.txt", "r") as file:
        lines = file.readlines()

    points = []

    for line in lines:
        x, y = map(int, line.strip().split(","))
        points.append((x, y))


    with open("result.txt", "w") as file:
        points = create_adjacency_matrix(points)
        for point in points:
             file.write(", ".join(map(str, point)) + "\n")

if __name__ == '__main__':
    main()