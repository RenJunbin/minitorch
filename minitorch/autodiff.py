from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    vals = list(vals)
    preresult = f(*vals)
    vals[arg] = vals[arg] + epsilon
    endresult = f(*vals)

    #print("central_difference is ", (endresult - preresult)/epsilon, preresult, endresult, epsilon)
    return (endresult - preresult)/epsilon

    #raise NotImplementedError('Need to implement for Task 1.1')


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    topsort = []
    tmpList = []
    permList = []

    def visit(var):
        if var.is_constant():
            #print("now, var is ", var.__dict__)
            return
        if var.unique_id in permList:
            return
        elif var.unique_id in tmpList:
            raise Exception("Not a DAG")
        
        tmpList.append(var.unique_id)
        if var.is_leaf():
            pass
        else:
            #parents = var.parents
            for i in var.history.inputs:
                visit(i)
        tmpList.remove(var.unique_id)
        permList.append(var.unique_id)
        topsort.insert(0, var)
    visit(variable)
    return topsort
    #raise NotImplementedError('Need to implement for Task 1.4')

def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    topsort = topological_sort(variable)
    derives = {variable.unique_id: deriv}
    #print("topsort is ", topsort)
    for node in topsort:
        if node.unique_id in derives:
            deriv = derives[node.unique_id]
        else:
            deriv = 0.0
        if node.is_leaf():
            node.accumulate_derivative(derives[node.unique_id])
        else:
            #print("derives is ", derives)
            for input, d in node.chain_rule(deriv):
                if input.unique_id not in derives:
                    derives[input.unique_id] = 0.0
                derives[input.unique_id] += d

    #raise NotImplementedError('Need to implement for Task 1.4')
    


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
