import utils
from PRP import implement_PRP
import numpy as np
from PRRP import implement_PRRP


if __name__ == '__main__':
    A, true_x, b = utils.generate_random_matrices(
        regression_matrix_rows=1000,
        regression_matrix_cols=120,
        density=0.9
    )
    x = np.zeros(true_x.shape)

    # Solve the problem using PRP
    calc_x, required_stage, L2_norm, R = implement_PRP(
        A=A, b=b, x=x, number_of_agents=6, required_L2_norm=0.5
    )

    calc_x = np.round(calc_x, 6)
    calc_b = np.matmul(A, calc_x)
    calc_b = np.round(calc_b, 6)

    with open('log.txt', 'w') as f:
        pass

    utils.write_to_file(true_x, calc_x, b, calc_b, R, required_stage, L2_norm, 'log.txt', algo_name='PRP')
    # PRP ends here

    # Solve the problem using PRRP
    calc_x, required_stage, L2_norm, R = implement_PRRP(
        A=A, b=b, x=x, number_of_agents=6, required_L2_norm=0.5
    )

    calc_x = np.round(calc_x, 6)
    calc_b = np.matmul(A, calc_x)
    calc_b = np.round(calc_b, 6)

    utils.write_to_file(true_x, calc_x, b, calc_b, R, required_stage, L2_norm, 'log.txt', algo_name='PRRP')
    # PRRP ends here