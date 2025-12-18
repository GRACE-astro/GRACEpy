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
                return self._print(base)
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

def print_matrix(expr,printer,outputs=["out"],mat_name="out",print_as_twod=False):
    lines = []         
    # Determine matrix shape
    rows, cols = expr.shape

    if len(outputs)==1:
        if rows == 1 or cols == 1:
            # --- 1D vector ---
            n = max(rows, cols)
            for i in range(n):
                lines.append(f"{mat_name}[{i}] = {printer.doprint(expr[i])};")

        else:
            if print_as_twod:
                if expr.is_symmetric() is True:
                    for i in range(rows):
                        for j in range(rows):
                            if i == j: lines.append(f"{mat_name}[{i}][{j}] = {printer.doprint(expr[i,j])};")
                            else: lines.append(f"{mat_name}[{i}][{j}] = {mat_name}[{j}][{i}] = {printer.doprint(expr[i,j])};")
                else: 
                    for i in range(rows):
                        for j in range(cols):
                            lines.append(f"{mat_name}[{i}][{j}] = {printer.doprint(expr[i,j])};")
            else:
                if expr.is_symmetric() is True:
                    ipos=0
                    for i in range(rows):
                        for j in range(i,rows):
                            lines.append(f"{mat_name}[{ipos}] = {printer.doprint(expr[i,j])};")
                            ipos+=1
                else: 
                    for i in range(rows):
                        for j in range(cols):
                            lines.append(f"{mat_name}[{j + cols * i}] = {printer.doprint(expr[i,j])};")
    else: 
        if rows == 1 or cols == 1:
            n = max(rows,cols)
            assert len(outputs) == n

            for i in range(n):
                lines.append(f"*{outputs[i]} = {printer.doprint(expr[i])};")
        else:
            raise ValueError("If more than one output is passed, expression must be a flat list")
    return '\n\t'.join(lines)


def make_body(exprs,printer,outputs=["out"],print_as_twod=False):
    printer = MyPrinter()
    subexprs, simplified_exprs = cse(exprs, optimizations='basic')
    
    optimized_sub = [(s[0], optimize(s[1], optims_c99)) for s in subexprs]

    lines = []
    for var, sub_expr in optimized_sub:
        lines.append(f'double {printer.doprint(Assignment(var, sub_expr))}')

    body = '\t' + '\n\t'.join(lines) + '\n\t'

    # now the output lines 
    if len(outputs) == len(simplified_exprs):
        for e,oo in zip(simplified_exprs,outputs):
            if isinstance(e, sp.Matrix):
                body += print_matrix(e,printer,["out"],oo,print_as_twod) + '\n\t' 
            else:
                body += f"*{oo} = " + printer.doprint(e) + ';\n\t'
    else:
        for e in simplified_exprs:
            if isinstance(e, sp.Matrix):
                body += print_matrix(e,printer,outputs,"out",print_as_twod) + '\n\t' 
            else:
                body += f"*out = " + printer.doprint(e) + ';\n\t'
    return body 

def base_name(s):
    name = str(s)
    if "[" in name:
        return name.split("[", 1)[0]  # take substring before first '['
    return name

def make_signature(exprs, ABI, name="compute_fluxes", outputs=["out"]):
    # collect all symbols used in all exprs
    all_symbols = set()
    for e in exprs:
        all_symbols.update(e.free_symbols)

    # sort for deterministic ordering
    all_symbols = sorted(all_symbols, key=lambda s: s.name)

    args = list()
    seen = list() 
    for s in all_symbols:
        n = base_name(s.name) 
        if n in ABI and not ( n in seen ):
            if (ABI[n].contains("[")):
                pass
            else:
                args.append(ABI[n] + f" {n}")
            seen.append(n)
        elif not n in seen: raise ValueError(f"Symbol {n} not found in ABI")
    for out_s in outputs:
        args.append(f"double* __restrict__ {out_s}")

    arg_str = ",\n\t".join(args)
    fsign = "static void KOKKOS_INLINE_FUNCTION\n"
    return f"{fsign}{name}(" + "\n\t" + f"{arg_str}" "\n)"

def make_function(exprs, printer, name, ABI, outputs=["out"],print_as_twod=False): 
    head = make_signature(exprs, ABI, name, outputs) 
    body = make_body(exprs,printer,outputs,print_as_twod)
    return head + "\n{\n" + body + "\n}\n"