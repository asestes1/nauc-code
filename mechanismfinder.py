import itertools
import gurobipy
import enum
import typing


class MechanismType(enum.Enum):
    DETERMINISTIC = enum.auto()
    PROBABILISTIC = enum.auto()

def has_best_item(item: int, valuation: tuple[int,...], congestion: tuple[int,...]):
    """
    Checks if an agent would benefit from deviating from its assigned item.
    item: the id of the item
    valuation: a tuple; valuation[i] is the base valuation of item i
    congestion: a tuple; congestion[i] is the congestion of item i
    """
    
    # print("CONGESTION:", congestion)
    # print("VALUATION: ", valuation)

    n_items = len(congestion)
    current_value = valuation[item] - congestion[item]
    for alt_item in range(n_items):
        if alt_item != item:
            alt_value = valuation[alt_item] - congestion[alt_item] - 1
            if alt_value > current_value:
                return False

    return True

def calc_congestion(assignment: tuple[int,...], n_items: int) -> tuple[int,...]:
    """
    Given an assignment, returns the congestion vector (congestion experienced by each item)
    assignment: a vector; assignment[i] is the item assigned to agent i
    n_items: the number of items
    """
    return tuple(sum(1 for assigned_item in assignment if item ==
                       assigned_item) for item in range(n_items))

def is_equilibrium(assignment: tuple[int,...], joint_valuation: tuple[tuple[int,...],...]) -> bool:
    """
    Checks if the assignment is an equilibrium for the given join_valuations
    assignment: a tuple; assignment[i] is the item assigned to agent i
    joint_valuation: a tuple of tuples; joint_valuation[i][j] is the base value that agent i gives item j
    """
    n_items = len(joint_valuation[0])
    congestion = calc_congestion(assignment, n_items)
    for participant, valuation in enumerate(joint_valuation):
        if not has_best_item(assignment[participant], valuation, congestion):
            return False
    return True

def generate_equilibria(joint_valuation: tuple[tuple[int,...],...]):
    """
    This generates all equilibria (allocations without singleton blocking coalitions)
    joint_valuation: tuple of tuples; joint_valuation[i][j] is agent's base valuation for item j
    """
    n_participants = len(joint_valuation)
    n_items = len(joint_valuation[0])

    for assignment in itertools.product(range(n_items), repeat=n_participants):
        if is_equilibrium(assignment, joint_valuation):
            yield assignment

def generate_neighborhood(participant, joint_valuation, permitted_valuations):
    new_joint_valuation = list(joint_valuation)
    for valuation in permitted_valuations:
        if valuation != joint_valuation[participant]:
            new_joint_valuation[participant] = valuation
            yield tuple(new_joint_valuation)


class ResultType(enum.Enum):
    FEASIBLE  = enum.auto()
    INFEASIBLE = enum.auto()

class InfeasProof(typing.NamedTuple):
    ir_duals: typing.Dict
    prob_duals: typing.Dict

class IcResult(typing.NamedTuple):
    resulttype: ResultType
    infeas_proof: typing.Optional[InfeasProof]
    icmech: typing.Optional[typing.Dict]

def find_ic_mechanism(n_participants: int, n_items: int, mechtype: MechanismType, permitted_valuations: frozenset[tuple[int,...]]):
    """
    This creates and solves a LP (or IP) to find an incentive-compatible mechanism that produces singleton-deviation-proof allocations
    n_participants: the number of participants
    n_items: the number of items
    mechtype: a MechanismType; either deterministic or probabilistic
    permitted_valuations: a set of vectors of possible valuations.
    """
    
    mymodel = gurobipy.Model()
    if mechtype == MechanismType.DETERMINISTIC:
        vartype = gurobipy.GRB.BINARY
    elif mechtype == MechanismType.PROBABILISTIC:
        vartype = gurobipy.GRB.CONTINUOUS
    else:
        raise ValueError("Specified mechanism type is not valid.")

    jv_generator = itertools.product(permitted_valuations, repeat=n_participants)
    all_eq = set()
    eq_sets = {}
    for jv in jv_generator:
        eq_sets[jv] = set(generate_equilibria(jv))
        for eq in eq_sets[jv]:
            all_eq.add(eq)
    
    congestions = {allocation: calc_congestion(allocation, n_items) for allocation in all_eq}
    myvars = {(jv, eq): mymodel.addVar(vtype=vartype, name="JV: "+str(jv)+", EQ: "+str(eq)) for jv, eq_set in eq_sets.items() for eq in eq_set}
    
    mymodel.Params.OutputFlag = 1
    mymodel.update()
    ir_constrs = {}
    for agent in range(0, n_participants):
        for jv, eq_set in eq_sets.items():
            rhs = gurobipy.LinExpr()
            for eq in eq_set:
                myrsrc = eq[agent]
                myutility = jv[agent][myrsrc] - congestions[eq][myrsrc]
                rhs.add(myvars[jv,eq], myutility)

            for alt_jv in generate_neighborhood(agent, jv, permitted_valuations):
                lhs = gurobipy.LinExpr()

                for alt_eq in eq_sets[alt_jv]:
                    myrsrc = alt_eq[agent]
                    myutility = jv[agent][myrsrc] - congestions[alt_eq][myrsrc]
                    lhs.add(myvars[alt_jv,alt_eq], myutility)
                ir_constrs[jv, alt_jv] = mymodel.addConstr(lhs <= rhs)

    prob_constrs={}
    for jv, eq_set in eq_sets.items():
        prob_constrs[jv] = mymodel.addConstr(gurobipy.quicksum(myvars[jv,eq] for eq in eq_set) == 1)
    mymodel.Params.InfUnbdInfo = 1
    mymodel.optimize()

    if mymodel.Status == gurobipy.GRB.INFEASIBLE:
        if mechtype == MechanismType.DETERMINISTIC:
            return IcResult(resulttype=ResultType.INFEASIBLE, infeas_proof=None, icmech=None)
        elif mechtype == MechanismType.PROBABILISTIC:
            ir_duals = {(jv, alt_jv): myconstr.FarkasDual for (jv, alt_jv), myconstr in ir_constrs.items() if abs(myconstr.FarkasDual) > 0.0001 }
            prob_duals = {jv: myconstr.FarkasDual for jv, myconstr in prob_constrs.items() if abs(myconstr.FarkasDual) > 0.0001 }
            return IcResult(resulttype=ResultType.INFEASIBLE, infeas_proof=InfeasProof(ir_duals=ir_duals, prob_duals=prob_duals), icmech=None)
    elif mymodel.Status == gurobipy.GRB.OPTIMAL:
        if mechtype == MechanismType.DETERMINISTIC:
            policy = {(jv, eq): myvar.X for (jv,eq), myvar in myvars.items() if myvar.X > 0.5}
        elif mechtype == MechanismType.PROBABILISTIC:
            policy = {(jv, eq): myvar.X for (jv,eq), myvar in myvars.items() if myvar.X > 0.0001}
            
        return IcResult(resulttype=ResultType.FEASIBLE, infeas_proof=None, icmech=policy)

    else:
        raise RuntimeError("Gurobi model is neither infeasible nor optimal. Not sure what happened!")
    


if __name__ == "__main__":
    permitted_valuations = frozenset({(3,4), (7,3), (3,7)})
    result = find_ic_mechanism(n_participants=4, n_items=2, mechtype=MechanismType.PROBABILISTIC, permitted_valuations=permitted_valuations)
    if result.resulttype == ResultType.INFEASIBLE:
        print("Infeasible.")
    else:
        print("A mechanism exists.")