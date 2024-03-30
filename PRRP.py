from utils import update_x_and_R
import numpy as np
import multiprocessing as mp
from RRP import implement_RRP


def implement_PRRP(A, b, x, number_of_agents, required_L2_norm):
    # Parallelized Random Residual Projection Algorithm

    p = number_of_agents
    A = np.split(A, p, axis=1)      # split the regression map A into "number_of_agents"
    R = b                           # divide b by p agents to get R
    w = 1/p                         # weight for each agent. for simplicity using equal values

    stage = 0
    while True:
        new_R = [R * w] * p         # divide R by the weights of p agents to get new_R

        # multiprocessing starts here
        parameter_list = [[i, j] for i, j in zip(A, new_R)]
        pool = mp.Pool(processes=p)
        result = pool.starmap(implement_RRP, parameter_list)
        pool.close()
        pool.join()
        # multiprocessing ends here

        x, R = update_x_and_R(x, result)
        L2_norm = np.linalg.norm(R)

        stage += 1
        if L2_norm <= required_L2_norm or stage >= 500:
            break

    return x, stage, L2_norm, R