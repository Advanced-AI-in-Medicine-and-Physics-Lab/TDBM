"L-system functions"


default = {
    "k": 3,
    "epsilon": 10,  # Proportion between length & diameter
    "randmarg": 3,  # Randomness margin between length & diameter
    "sigma": 5,  # Determines type deviation for Gaussian distributions
    "d": 2,
    "stochparams": True,
}  # Whether the generated parameters will also be stochastic


def setProperties(properties):
    """
    Establishes property values according to a dictionary given as input
    """
    if properties == None:
        properties = default

    global k, epsilon, randmarg, sigma, d, stochparams

    k = properties["k"]
    epsilon = properties["epsilon"]
    randmarg = properties["randmarg"]
    sigma = properties["sigma"]
    d = properties["d"]
    stochparams = properties["stochparams"]
