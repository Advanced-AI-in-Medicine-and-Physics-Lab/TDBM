import random

import numpy as np
from .analyseGrammar import posneg
from .libGenerator import calBifurcation, calParam, setProperties
def carotid_F(n, d0):
    """
    Generates a Carotid Artery System with robust string formatting.
    """
    # 1. Common Carotid Trunks (Long)
    # Force val to be float to avoid division by zero or type errors
    trunk = ""
    for _ in range(4):
        trunk += D_carotid(d0, length_factor=2.0)

    # 2. Bifurcation Logic
    # ICA ~70%, ECA ~60% of CCA diameter
    d_ica = d0 * 0.70
    d_eca = d0 * 0.60
    
    theta_ica = np.random.uniform(5, 12)  # Narrow
    theta_eca = np.random.uniform(25, 45) # Wider
    
    # Explicit formatting to prevent parser errors
    bifurcation = (
        "[" 
        + f"+({theta_ica:.2f})" 
        + f"/({10.0:.2f})" 
        + carotid_branch_ICA(n, d_ica) 
        + "]"
        + "[" 
        + f"-({theta_eca:.2f})" 
        + f"/({-10.0:.2f})" 
        + carotid_branch_ECA(n, d_eca) 
        + "]"
    )
    
    return trunk + bifurcation

def D_carotid(d0, length_factor=2.0):
    params = calBifurcation(d0)
    # Calculate length: 'co' is characteristic length from libGenerator
    length = params["co"] / float(length_factor)
    return f"f({length:.3f},{d0:.3f})"

def carotid_branch_ICA(n, d):
    if n <= 0: return f"f(1.0,{d:.3f})"
    # Long, smooth, tapering slightly
    return f"f(5.0,{d:.3f})" + carotid_branch_ICA(n - 1, d * 0.99)

def carotid_branch_ECA(n, d):
    if n <= 0: return f"f(1.0,{d:.3f})"
    
    # ECA has side branches
    has_branch = random.random() < 0.5
    main_vessel = f"f(3.0,{d:.3f})"
    
    if has_branch:
        branch_d = d * 0.45
        branch_angle = np.random.uniform(45, 80)
        # Side branch
        side = f"[" + f"+({branch_angle:.2f})" + f"f(2.0,{branch_d:.3f})" + "]"
        return main_vessel + side + carotid_branch_ECA(n - 1, d * 0.96)
    else:
        return main_vessel + carotid_branch_ECA(n - 1, d * 0.96)
    
def I(n, d0, val=3):
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join("co/", str(int(val))), params)
        rotate = np.random.uniform(22.5, 27.5)
        return (
            "f("
            + p1
            + ","
            + str(params["d0"])
            + ")"
            + "+("
            + str(rotate)
            + ")"
            + "["
            + R(n - 1, params["d0"])
            + "]"
        )
    else:
        return "I"


def R(n, d0):
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join("co/", str(int(3))), params)
        p2 = calParam(str.join("co/", str(int(2))), params)
        descrip = (
            "f("
            + p1
            + ")"
            + G(n, d0, val=7)
            + G(n - 1, d0, val=7)
            + G(n - 1, d0, val=7)
            + "["
            + B(n - 1, params["d1"])
            + "]"
            + "f("
            + p2
            + ","
            + str(params["d2"])
            + ")"
            + B(n - 1, params["d2"])
        )
        return descrip
    else:
        return "R"


def B(n, d0):
    if n > 0:
        return (
            G(n - 1, d0, val=7)
            + G(n - 1, d0, val=7)
            + G(n - 1, d0, val=7)
            + "/("
            + str(90.0)
            + ")"
            + A(n, d0)
        )
    else:
        return "B"

def coronary_F(niter, d0, artery_type="LCA"):
    """Build a coronary-artery L-system string by recursive rule expansion."""
    if artery_type in ("LCA", "RCA"):
        base_rules = "F[+F][-F]"  # branch up and down from each segment
    else:
        raise ValueError("Unknown artery type: {}".format(artery_type))

    rule = base_rules
    for _ in range(niter):
        rule = rule.replace("F", base_rules)
    return rule



def F(n, d0):
    if n > 0:
        params = calBifurcation(d0)
        theta1 = params["th1"]  # + np.random.uniform(-2.5, 2.5)
        theta2 = params["th2"]  # + np.random.uniform(-2.5, 2.5)
        tilt = np.random.uniform(22.5, 27.5) * random.randint(-1, 1)

        return (
            S(n - 1, d0)
            + "["
            + "+("
            + str(theta1)
            + ")"
            + "/("
            + str(tilt)
            + ")"
            + F(n - 1, params["d1"])
            + "]"
            + "["
            + "-("
            + str(theta2)
            + ")"
            + "/("
            + str(tilt)
            + ")"
            + F(n - 1, params["d2"])
            + "]"
        )
    else:
        return "F"


def S(n, d0, val=5, margin=0.5):
    r = random.random()
    if r >= 0.0 and r < margin:
        return "{" + S1(n, d0, val) + "}"
    if r >= margin and r < 1.0:
        return "{" + S2(n, d0, val) + "}"


def S1(n, d0, val=5):
    if n > 0:
        # "Fanning" of trees
        rotate = np.random.uniform(22.5, 27.5) * random.randint(-1, 1)
        params = calBifurcation(d0)
        descrip = (
            D(n - 1, params["d0"], val)
            + "+("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
            + "-("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
            + "-("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
            + "+("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
        )
        return descrip
    else:
        return "S"


def S2(n, d0, val=5):
    if n > 0:
        # "Fanning" of trees
        rotate = np.random.uniform(22.5, 27.5) * random.randint(-1, 1)
        params = calBifurcation(d0)
        descrip = (
            D(n - 1, params["d0"], val)
            + "-("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
            + "+("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
            + "+("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
            + "-("
            + str(rotate)
            + ")"
            + D(n - 1, params["d0"], val)
        )
        return descrip
    else:
        return "S"


def D(n, d0, val=5):
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join("co/", str(int(val))), params)
        return "f(" + p1 + "," + str(params["d0"]) + ")"
    else:
        return "D"


def G(n, d0, val=5, shift=18.0):
    if n > 0:
        params = calBifurcation(d0)
        p1 = calParam(str.join("co/", str(int(val))), params)
        return "f(" + p1 + "," + str(params["d0"]) + ")"
    else:
        return "G"


def A(n, d0):
    if n > 0:
        params = calBifurcation(d0)
        return (
            S(n - 1, d0)
            + "[+("
            + str(params["th1"])
            + ")"
            + A(n - 1, params["d1"])
            + "]"
            + "[+("
            + str(params["th2"])
            + ")"
            + A(n - 1, params["d2"])
        )
    else:
        return "A"


# def A(d0):
#     return 'f(' + str(1) + ',' + str(d0) + ')' + 'f(' + str(1) + ',' + str(2*d0) + ')' + \
#         'f(' + str(1) + ',' + str(d0) + ')'

# def E(d0):
#     return 'f(' + str(1) + ',' + str(d0) + ')' + 'f(' + str(1) + ',' + str(0.5*d0) + ')' + \
#         'f(' + str(1) + ',' + str(d0) + ')'


# 2D grammars - don't currently work with framework
def simplest_gramma(n=None, theta1=20.0, theta2=20.0, params=None):
    return "f" + "[" + "+" + F(n - 1, params["d0"]) + "]" + "-" + F(n - 1, params["d0"])


def simple_grammar(n=None, theta1=20.0, theta2=20.0, params=None):
    return (
        "f"
        + "["
        + "+("
        + str(theta1)
        + ")"
        + F(n - 1, params["d1"])
        + "]"
        + "["
        + "-("
        + str(theta2)
        + ")"
        + F(n - 1, params["d2"])
        + "]"
    )
