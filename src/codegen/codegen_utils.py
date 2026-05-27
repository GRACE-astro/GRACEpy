from sympy import symbols, Rational
from sympy.utilities.codegen import codegen
from sympy.codegen.rewriting import optimize, optims_c99
from sympy.simplify.cse_main import cse
import sympy as sp
from sympy import S
from sympy.printing.c import C99CodePrinter, Assignment
try:
    from sympy.printing.numpy import NumPyPrinter            # sympy >= 1.11
except ImportError:                                           # pragma: no cover
    from sympy.printing.pycode import NumPyPrinter            # older sympy
from sympy import init_printing

class MyPrinter(C99CodePrinter):
    def _print_Pow(self, expr):
        base, exp = expr.as_base_exp()

        # Only expand integer powers >= 0
        if exp.is_integer:
            exp_val = int(exp)
            if exp_val == 0:
                return "1"
            if exp_val == 1:
                return "("+self._print(base)+")"
            # Emit base*base*base...
            if exp.is_nonnegative: return "("+ "*".join(["("+self._print(base)+")"] * exp_val) + ")"
            else: return f"1/({self._print(base**abs(exp))})"

        # handle x**(-1/2)
        if expr.exp == -sp.Rational(1,2):
            return f"(1.0/sqrt({self._print(expr.base)}))"

        # handle x**(-3/2)
        if expr.exp == -sp.Rational(3,2):
            base = f"(1.0/sqrt({self._print(expr.base)}))"
            return f"({base}*{base}*{base})"

        # Otherwise fallback to regular pow(...)
        return super()._print_Pow(expr)

    def _print_Piecewise(self, expr):
        # Emit Piecewise as a nested C ternary so it can appear on the RHS of
        # a `double x = ...;` line produced by make_body. Requires the last
        # branch to be a catch-all (cond == True); otherwise fall back to the
        # parent's if/else statement form.
        if expr.args[-1].cond != sp.S.true:
            return super()._print_Piecewise(expr)
        *branches, default = expr.args
        result = f"({self._print(default.expr)})"
        for e, c in reversed(branches):
            result = f"(({self._print(c)}) ? ({self._print(e)}) : {result})"
        return result

    def _print_Assignment(self, expr):
        # Bypass the parent's Piecewise -> if/else special case: we want
        # inline ternaries so `double x = <piecewise>;` stays a single
        # statement. Non-Piecewise RHS falls through to the default path.
        from sympy.functions.elementary.piecewise import Piecewise
        if isinstance(expr.rhs, Piecewise):
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return self._get_statement(f"{lhs} = {rhs}")
        return super()._print_Assignment(expr)

def derivative_matrix(mat, name, deriv_suffix):
    rows, cols = mat.shape
    out = sp.MutableDenseMatrix(rows, cols, [0]* (rows*cols))

    for i in range(rows):
        for j in range(cols):
            out[i,j] = sp.symbols(f"{name}_{deriv_suffix}[{i}][{j}]")
    return out

def der_symm_tens(mat,name):
    rows, cols = mat.shape
    if rows != cols: raise ValueError("Non-square tensor cannot be symmetric")
    n = rows 
    ncomps = n * (n+1)//2 
    out = [] 
    for idir in range(3):
        _out = sp.MutableDenseMatrix(n, n, [0]* (n*n))
        icomp = 0 
        for i in range(n):
            for j in range(i,n):
                # order idir, i, j
                _out[i,j] = _out[j,i] = sp.symbols(f"d{name}_dx[{icomp + ncomps * idir}]")
                icomp += 1
        out.append(_out)
    return out

def derivative_vector(vec, name, deriv_suffix):
    n = vec.shape[0]
    out = sp.MutableDenseMatrix([0]*n)
    for i in range(n):
        out[i] = sp.symbols(f"{name}_{deriv_suffix}[{i}]")
    return out

def der_vec(vec,name):
    out = [] 
    n = vec.shape[0]
    for idir in range(3):
        _out = sp.Matrix([0]*n)
        for i in range(n):
            _out[i] = sp.symbols(f"d{name}_dx[{ i + n * idir}]")
        out.append(_out)
    return out

def emit_matrix_assignments(expr, printer, name, layout="flat", enforce_symmetry=True, addto=False):
    rows, cols = expr.shape
    lines = []

    is_vector = (rows == 1 or cols == 1)
    is_symmetric = bool(expr.is_symmetric()) if enforce_symmetry else False

    voigt_map = [(i,j) for i in range(rows) for j in range(i, cols)]

    if is_vector:
        n = max(rows, cols)
        for i in range(n):
            idx = i if layout == "flat" else f"{i}"
            if addto:
                lines.append(f"(*{name})[{idx}] += {printer.doprint(expr[i])};")
            else:
                lines.append(f"(*{name})[{idx}] = {printer.doprint(expr[i])};")
        return lines

    if is_symmetric:
        if layout == "flat":
            for idx, (i,j) in enumerate(voigt_map):
                val = printer.doprint(expr[i, j])
                if addto:
                    lines.append(f"(*{name})[{idx}] += {val};")
                else:
                    lines.append(f"(*{name})[{idx}] = {val};")
        else:
            for i in range(rows):
                for j in range(i, cols):
                    val = printer.doprint(expr[i, j])
                    if i == j:
                        if addto:
                            lines.append(f"(*{name})[{i}][{j}] += {val};")
                        else:
                            lines.append(f"(*{name})[{i}][{j}] = {val};")
                    else:
                        if addto: raise ValueError("Cannot add to output in extended layout")
                        lines.append(
                            f"(*{name})[{i}][{j}] = (*{name})[{j}][{i}] = {val};"
                        )
    else:
        for i in range(rows):
            for j in range(cols):
                val = printer.doprint(expr[i, j])
                if layout == "flat":
                    idx = j + cols*i
                    if addto:
                        lines.append(f"(*{name})[{idx}] += {val};")
                    else:
                        lines.append(f"(*{name})[{idx}] = {val};")
                else:
                    if addto:
                        lines.append(f"(*{name})[{i}][{j}] += {val};")
                    else:
                        lines.append(f"(*{name})[{i}][{j}] = {val};")

    return lines

def emit_output(expr, printer, out_name, layout="flat", addto=False):
    if isinstance(expr, sp.Matrix):
        return emit_matrix_assignments(expr, printer, out_name, layout, addto=addto)
    else:
        if addto:
            return [f"*{out_name} += {printer.doprint(expr)};"]
        else:
            return [f"*{out_name} = {printer.doprint(expr)};"]

def make_body(exprs, printer, outputs, layout="flat", cse_order="canonical" ,cse_optims='basic', cse_ignore=(), addto=False):
    subexprs, reduced = cse(exprs, optimizations=cse_optims, order=cse_order, ignore=cse_ignore)

    lines = []

    # temporaries
    for var, sub in subexprs:
        if ( cse_optims == 'basic' and isinstance(sub, sp.Expr) ): sub = optimize(sub, optims_c99)
        lines.append(f"double {printer.doprint(Assignment(var, sub))}")

    # outputs
    if len(outputs) == len(reduced):
        for expr, name in zip(reduced, outputs):
            lines.extend(emit_output(expr, printer, name, layout, addto))
    else:
        for expr in reduced:
            lines.extend(emit_output(expr, printer, outputs[0], layout, addto))

    return "\t" + "\n\t".join(lines)

def format_arg(name, abi):
    ctype, shape = abi
    if shape is None:
        return f"{ctype} {name}"
    else:
        dims = "".join(f"[{n}]" for n in shape)
        return f"const {ctype} {name}{dims}"

def format_output(name,abi):
    ctype, shape = abi
    if shape is None:
        return f"{ctype} * __restrict__ {name}"
    else:
        dims = "".join(f"[{n}]" for n in shape)
        return f"{ctype} (*{name}){dims}"

def base_name(s):
    name = str(s)
    if "[" in name:
        return name.split("[", 1)[0]  # take substring before first '['
    elif "(" in name:
        return name.split("(", 1)[0]  # take substring before first '['
    return name

from collections import OrderedDict

def generate_signature(
    name,
    exprs,
    additional_inputs,
    outputs,
    ABI,
    outputs_ABI,
    format_arg,
    format_output,
    template_args=None,
    global_constants=[]
):
    """
    Generate a stable function signature with arguments ordered
    according to ABI and outputs_ABI.
    """

    # ----------------------------------------------------------------------
    # 1. Build ABI order maps for fast integer ordering
    # ----------------------------------------------------------------------
    ABI_order_map = {k: i for i, k in enumerate(ABI.keys())}
    outputs_order_map = {k: i for i, k in enumerate(outputs_ABI.keys())}

    # ----------------------------------------------------------------------
    # 2. Collect all required argument names (inputs + additional inputs)
    # ----------------------------------------------------------------------
    input_syms = set()

    # Symbols from expression free symbols
    for e in exprs:
        for s in e.free_symbols:
            n = base_name(s)
            input_syms.add(n)

    # Manually declared additional inputs
    for s in additional_inputs:
        n = base_name(s)
        input_syms.add(n)

    # Outputs must remain in their own ordered class
    output_syms = [base_name(o) for o in outputs]

    # Remove outputs from input list (outputs appear separately)
    input_syms = [s for s in input_syms if s not in output_syms]

    # ----------------------------------------------------------------------
    # 3. ABI-sorted ordering
    # ----------------------------------------------------------------------
    # Inputs: sorted by ABI order, falling back to alphabetical
    def input_key(n):
        if n in ABI_order_map:
            return (0, ABI_order_map[n])   # primary: ABI order
        return (1, n)                      # secondary: alphabetical fallback

    input_syms_sorted = sorted(input_syms, key=input_key)

    # Outputs: strictly ABI order
    def output_key(n):
        if n not in outputs_order_map:
            raise ValueError(f"Output symbol {n} missing from outputs_ABI")
        return outputs_order_map[n]

    output_syms_sorted = sorted(output_syms, key=output_key)

    # ----------------------------------------------------------------------
    # 4. Format arguments
    # ----------------------------------------------------------------------
    args = []

    # Inputs
    for n in input_syms_sorted:
        if n not in ABI and n not in global_constants:
            raise ValueError(f"Symbol {n} missing from ABI")
        args.append(format_arg(n, ABI[n]))

    # Outputs
    for o in output_syms_sorted:
        args.append(format_output(o, outputs_ABI[o]))

    # ----------------------------------------------------------------------
    # 5. Build final signature string
    # ----------------------------------------------------------------------
    if template_args is None:
        sig = (
            "static void KOKKOS_INLINE_FUNCTION\n"
            f"{name}(\n\t" + ",\n\t".join(args) + "\n)"
        )
    else:
        templates = [f"{t_type} {t_name}" for (t_type, t_name) in template_args]
        sig = (
            "template< " + ", ".join(templates) + " >\n"
            "static void KOKKOS_INLINE_FUNCTION\n"
            f"{name}(\n\t" + ",\n\t".join(args) + "\n)"
        )

    return sig


def make_function(exprs, printer, name, ABI, outputs, outputs_ABI, layout="flat", additional_inputs=[], cse_order='canonical', cse_optims='basic', template_args=None, global_constants=[], cse_ignore=(), add_to_output=False):
    sig = generate_signature(name,exprs,additional_inputs,outputs,ABI,outputs_ABI,format_arg,format_output,template_args,global_constants)

    body = make_body(exprs, printer, outputs, layout, cse_order, cse_optims, cse_ignore, add_to_output)

    return sig + "\n{\n" + body + "\n}\n"


# ============================================================================
# Python emitter: parallel of make_function that produces a numpy-flavoured
# Python function with the same name, ABI ordering, and CSE structure as the
# C version.  Useful for stress-testing solvers in isolation without rebuilding
# the full numerical-relativity codebase.
# ============================================================================

class MyPyPrinter(NumPyPrinter):
    """NumPy printer that emits Piecewise as inline `a if c else b` ternaries
    so a CSE temporary `x = <piecewise>` is one statement, mirroring how
    MyPrinter handles the C ternary.  Relational operators are emitted as
    plain Python operators (==, >=, …) rather than numpy.equal/etc., which
    keeps the generated code readable and avoids module-alias collisions."""

    _module_format_overrides = {"numpy.sqrt": "np.sqrt", "numpy.array": "np.array"}

    def _module_format(self, fqn, register=True):
        # Force every numpy.* spelling to np.* so the generated code matches
        # `import numpy as np` at the top of the emitted module.
        return super()._module_format(fqn, register).replace("numpy.", "np.")

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != sp.S.true:
            return super()._print_Piecewise(expr)
        *branches, default = expr.args
        result = f"({self._print(default.expr)})"
        for e, c in reversed(branches):
            result = f"({self._print(e)} if {self._print(c)} else {result})"
        return result

    # Plain Python relationals — needed because Piecewise conditions are
    # always scalar at call time (idir, sign(Bn), etc.) so we don't want
    # vectorized numpy.equal/numpy.greater_equal calls there.
    def _print_Equality(self, expr):
        return f"({self._print(expr.args[0])} == {self._print(expr.args[1])})"
    def _print_Unequality(self, expr):
        return f"({self._print(expr.args[0])} != {self._print(expr.args[1])})"
    def _print_StrictGreaterThan(self, expr):
        return f"({self._print(expr.args[0])} > {self._print(expr.args[1])})"
    def _print_GreaterThan(self, expr):
        return f"({self._print(expr.args[0])} >= {self._print(expr.args[1])})"
    def _print_StrictLessThan(self, expr):
        return f"({self._print(expr.args[0])} < {self._print(expr.args[1])})"
    def _print_LessThan(self, expr):
        return f"({self._print(expr.args[0])} <= {self._print(expr.args[1])})"

    def _print_Pow(self, expr):
        # Special-case x**(-1/2) to a sqrt for parity with MyPrinter; let the
        # parent handle integer powers (Python's ** is fine) and other cases.
        if expr.exp == -sp.Rational(1, 2):
            return f"(1.0/np.sqrt({self._print(expr.base)}))"
        if expr.exp == -sp.Rational(3, 2):
            base = f"(1.0/np.sqrt({self._print(expr.base)}))"
            return f"({base}*{base}*{base})"
        return super()._print_Pow(expr)

    # NumPyPrinter routes Max/Min through functools.reduce(numpy.maximum,...);
    # emit np.maximum.reduce(...) directly so the generated module needs only
    # `import numpy as np`.
    def _print_Max(self, expr):
        if len(expr.args) == 2:
            return f"np.maximum({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        elems = ", ".join(self._print(a) for a in expr.args)
        return f"np.maximum.reduce([{elems}])"
    def _print_Min(self, expr):
        if len(expr.args) == 2:
            return f"np.minimum({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        elems = ", ".join(self._print(a) for a in expr.args)
        return f"np.minimum.reduce([{elems}])"


def _emit_output_py(expr, printer, out_name, layout="flat"):
    """Build a Python statement (or block of statements) that assigns the
    SymPy expression to `out_name`.  Vectors are emitted as np.array([...]),
    matrices as np.array([...]).reshape((rows, cols)) when layout=="flat",
    or as a 2-D array literal otherwise."""
    if isinstance(expr, sp.Matrix):
        rows, cols = expr.shape
        if rows == 1 or cols == 1:
            n = max(rows, cols)
            elems = ", ".join(printer.doprint(expr[i]) for i in range(n))
            return [f"{out_name} = np.array([{elems}])"]
        elems = ", ".join(
            printer.doprint(expr[i, j]) for i in range(rows) for j in range(cols)
        )
        if layout == "flat":
            return [f"{out_name} = np.array([{elems}])"]
        return [f"{out_name} = np.array([{elems}]).reshape(({rows}, {cols}))"]
    return [f"{out_name} = {printer.doprint(expr)}"]


def _make_body_py(exprs, printer, outputs, layout, cse_order, cse_optims, cse_ignore):
    subexprs, reduced = cse(exprs, optimizations=cse_optims,
                            order=cse_order, ignore=cse_ignore)
    lines = []
    for var, sub in subexprs:
        if cse_optims == 'basic' and isinstance(sub, sp.Expr):
            sub = optimize(sub, optims_c99)
        lines.append(f"{printer.doprint(var)} = {printer.doprint(sub)}")
    if len(outputs) == len(reduced):
        for expr, name in zip(reduced, outputs):
            lines.extend(_emit_output_py(expr, printer, name, layout))
    else:
        for expr in reduced:
            lines.extend(_emit_output_py(expr, printer, outputs[0], layout))
    if len(outputs) == 1:
        lines.append(f"return {outputs[0]}")
    else:
        lines.append(f"return ({', '.join(outputs)})")
    return "    " + "\n    ".join(lines)


def make_function_py(exprs, printer, name, ABI, outputs, outputs_ABI,
                     layout="flat", additional_inputs=(),
                     cse_order='canonical', cse_optims='basic',
                     cse_ignore=(), global_constants=()):
    """Python counterpart of make_function.  Emits a `def name(args): ...`
    block whose argument order matches the C ABI ordering exactly.  Outputs
    are returned as a tuple (or as a single value when len(outputs)==1).

    Argument names are passed through the printer's reserved-keyword escape
    (e.g. `lambda` → `lambda_`) so the signature matches the body, where
    sympy applies the same escape automatically."""

    ABI_order = {k: i for i, k in enumerate(ABI.keys())}

    input_syms = set()
    for e in exprs:
        for s in e.free_symbols:
            input_syms.add(base_name(s))
    for s in additional_inputs:
        input_syms.add(base_name(s))

    output_syms = [base_name(o) for o in outputs]
    input_syms = [s for s in input_syms if s not in output_syms]

    def key(n):
        if n in ABI_order:
            return (0, ABI_order[n])
        return (1, n)

    args_raw = sorted(input_syms, key=key)

    for n in args_raw:
        if n not in ABI and n not in global_constants:
            raise ValueError(f"Symbol {n} missing from ABI")

    # Escape Python reserved words to match what the printer emits in the body
    # (sympy's PythonCodePrinter rewrites e.g. `lambda` → `lambda_`).
    reserved   = getattr(printer, 'reserved_words', set())
    suffix     = printer._settings.get('reserved_word_suffix', '_') if hasattr(printer, '_settings') else '_'
    def escape(n):
        return n + suffix if n in reserved else n
    args = [escape(n) for n in args_raw]

    sig  = f"def {name}({', '.join(args)}):"
    body = _make_body_py(exprs, printer, output_syms, layout,
                         cse_order, cse_optims, cse_ignore)
    return sig + "\n" + body + "\n"

