import numpy as np
from PRP import implement_PRP

class SimpleAgent:
    def __init__(self, a, b, sigma_w, T):
        """
        Initializes an agent with its own 1D dynamical system.
        :param a: System parameter.
        :param b: Input parameter.
        :param sigma_w: Standard deviation of process noise.
        :param T: Number of time steps for data generation.
        """
        self.a = a
        self.b = b
        self.sigma_w = sigma_w
        self.T = T
        self.x = np.zeros(T + 1)
        self.u = np.random.uniform(-1, 1, T)

    def generate_data(self):
        """
        Generates data by simulating the dynamical system.
        """
        for t in range(self.T):
            w = np.random.normal(0, self.sigma_w)  # Process noise
            self.x[t + 1] = self.a * self.x[t] + self.b * self.u[t] + w

    def get_data(self):
        """
        Returns the generated data.
        """
        return self.x, self.u
if __name__ == "__main__":
    # Parameters for simulation
    T = 500  # Time steps
    sigma_w = 0.1  # Standard deviation of process noise

    # Creating two agents with slightly different system parameters
    agent1 = SimpleAgent(a=1.0, b=1.0, sigma_w=sigma_w, T=T)
    agent2 = SimpleAgent(a=1.05, b=1.5, sigma_w=sigma_w, T=T)

    # Generate and observe data for both agents
    agent1.generate_data()
    agent2.generate_data()

    # Retrieving and printing the first few data points as a simple check
    x1, u1 = agent1.get_data()
    x2, u2 = agent2.get_data()

    print("Agent 1 data (first 5 states):", x1[:5])
    print("Agent 2 data (first 5 states):", x2[:5])


    # Combine the observations from both agents
    A_agent1 = np.vstack([x1[:-1], u1]).T  # Observations for Agent 1
    b_agent1 = x1[1:]  # Targets for Agent 1

    A_agent2 = np.vstack([x2[:-1], u2]).T  # Observations for Agent 2
    b_agent2 = x2[1:]  # Targets for Agent 2

    # Stack the observations and targets from both agents
    A = np.vstack([A_agent1, A_agent2])
    b = np.hstack([b_agent1, b_agent2])

    #Initial guess for the parameters
    x = np.zeros(2)
    # Solve the problem using PRP
    calc_x, required_stage, L2_norm, R = implement_PRP(
        A=A, b=b, x=x, number_of_agents=2, required_L2_norm=0.5
    )
    

    print("Estimated parameters (x) using PRP:", calc_x)


