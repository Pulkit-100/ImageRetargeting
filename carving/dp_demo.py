import random
import sys


def pprint(A):
    for i in A:
        print(*i)

    print("\n\n")


def generate_matrix():
    n = random.randint(3,6)
    m = random.randint(3,6)

    matrix = [[None for i in range(m)] for j in range(n)]

    for i in range(n):
        for j in range(m):
            matrix[i][j] = random.randint(0,3)

    return n, m, matrix


def fill_dp(energyMatrix, i, j, n, m, dp):

    if i == n:
        return 0

    if j < 0 or j >= m:
        return sys.maxsize

    if (i, j) in dp:
        return dp[(i, j)]

    dp[(i, j)] = energyMatrix[i][j] + min([fill_dp(energyMatrix, i+1, j, n, m, dp),
                                           fill_dp(energyMatrix, i + 1, j+1, n, m, dp),
                                           fill_dp(energyMatrix, i + 1, j-1, n, m, dp)])

    # cummulativePath[i][j] = energyMatrix[i][j] + min(cummulativePath[i+1][j-1],
    #                                                  cummulativePath[i+1][j],
    #                                                  cummulativePath[i+1][j+1])

    return dp[(i, j)]


def remove_and_display_seam(processedPath, energyMatrix, i, j, n, m):

    print(" "*(2*j) + str(energyMatrix[i][j]))

    if i < n-1:

        mini = min(processedPath[i+1][max(0, j-1):min(j+2, m)])

        for k in range(max(0, j-1),min(j+2, m)):
            if processedPath[i+1][k] == mini:
                remove_and_display_seam(processedPath, energyMatrix, i+1, k, n, m)
                break

    energyMatrix[i].pop(j)


if __name__ == '__main__':

    n, m, energyMatrix = generate_matrix()

    x = m//2
    for i in range(x):  # Removing Seams half the width

        print("Energy Matrix \n")
        pprint(energyMatrix)

        dp = {}  # dp[i,j] = lowest energy required to reach last row from ith row and jth column

        for j in range(m):
            fill_dp(energyMatrix, 0, j, n, m, dp)

        processedPath = [[dp[(i, j)] for j in range(m)] for i in range(n)]

        print("Processed Path DP matrix \n")
        pprint(processedPath)

        index = processedPath[0].index(min(processedPath[0]))

        print("Seam Removed \n")
        remove_and_display_seam(processedPath, energyMatrix, 0, index, n, m)
        m -= 1

        print("\n\n")

    else:
        print("Energy Matrix \n")
        pprint(energyMatrix)
