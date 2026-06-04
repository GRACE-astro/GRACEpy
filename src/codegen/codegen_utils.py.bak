from sympy import symbols, Rational
from sympy.utilities.codegen import codegen
from sympy.codegen.rewriting import optimize, optims_c99
from sympy.simplify.cse_main import cse
import sympy as sp
from sympy import S
from sympy.printing.c import C99CodePrinter, Assignment
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
        if ( cse_optims == 'basic' ): sub = optimize(sub, optims_c99)
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

