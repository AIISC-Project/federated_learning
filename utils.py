import numpy  as np
from scipy import sparse


random_seed = 42
np.random.seed(random_seed)


def generate_random_matrices(regression_matrix_rows=1000,
                             regression_matrix_cols=840,
                             density=0.1):
    # Generates random regression map and observations
    # A = regression map
    # z = original x ---> we are trying to find this one
    # b = observation vector

    A = np.random.uniform(low=0, high=1, size=(regression_matrix_rows, regression_matrix_cols))
    z = sparse.random(regression_matrix_cols, 1, density=density).toarray()
    b = np.matmul(A, z)

    return A, z, b


def sub_problem(A, b):
    # solve the sub-problem for each agent

    x = np.matmul(np.linalg.pinv(A), b)
    r = b - np.matmul(A, x)

    return (x, r)


def update_x_and_R(x, result):
    # update x from first element of result
    new_x = []
    for i in result:
        for j in i[0]:
            new_x.append(j.tolist())

    x = x + new_x
    # update of x done

    # update R from second element of result
    new_R = None
    for i in result:
        if new_R is None:
            new_R = i[1].tolist()
        else:
            new_R = np.array(new_R) + np.array(i[1].tolist())
    # update of R done

    return x, new_R


def get_probability_distribution(A):
    # get probability distribution of each column of A

    transposed_A = np.transpose(A).tolist()
    probability_distribution = []

    for i in transposed_A:
        probability_distribution.append(np.linalg.norm(i) / np.linalg.norm(A))

    normalized_probability_distribution = probability_distribution / np.sum(probability_distribution)

    return normalized_probability_distribution


def write_to_file(true_x, calc_x, b, calc_b, R, required_stage, L2_norm, file_name, algo_name):
    # write the output to a file

    with open(file_name, 'a') as f:
        f.write(f'\n\nOutput for {algo_name}\n')

        f.write('\nTrue x:\n')
        for e in true_x:
            f.write(f'{e}')

        f.write('\nCalculated x:\n')
        for e in calc_x:
            f.write(f'{e}')

        f.write('\nOriginal b:\n')
        for e in b:
            f.write(f'{e} ')

        f.write('\nCalculated b:\n')
        for e in calc_b:
            f.write(f'{e} ')

        f.write('\nR:\n')
        for e in R:
            f.write(f'{e} ')

        f.write(f'\nRequired Stages:\n {required_stage}')
        f.write(f'\nL2 Norm\n: {L2_norm}\n')

    # write the output to a file ends here