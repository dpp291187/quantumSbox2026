from math import log2
from statistics import median
from collections import defaultdict, Counter

# =========================================================
# 0) Optional Qiskit import
# =========================================================
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import XGate
    from qiskit.converters import circuit_to_dag
    HAS_QISKIT = True
except ImportError:
    QuantumCircuit = None
    transpile = None
    XGate = None
    circuit_to_dag = None
    HAS_QISKIT = False


# =========================================================
# 1) Helpers
# =========================================================
def popcount(x):
    """Return the Hamming weight of an integer."""
    return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")


def agg_min_med_max(xs):
    """Return min / median / max for a list of numbers."""
    if not xs:
        return {"min": None, "med": None, "max": None}
    return {
        "min": float(min(xs)),
        "med": float(median(xs)),
        "max": float(max(xs)),
    }


def is_bijective(S):
    """Check whether the S-box is bijective."""
    return len(set(S)) == len(S)


def infer_nm(S):
    """
    Infer:
      n = number of input bits
      m = number of output bits
    from the S-box lookup table.
    """
    L = len(S)
    n = int(round(log2(L)))
    if (1 << n) != L:
        raise ValueError("len(S) must be 2^n.")

    vmax = max(S) if S else 0
    m = max(1, vmax.bit_length())

    if any(v < 0 or v >= (1 << m) for v in S):
        raise ValueError("S-box values are out of range for the inferred m.")

    return n, m


# =========================================================
# 2) Möbius transform -> ANF
# =========================================================
def mobius_anf(f_vals, n):
    """
    Convert a truth table (0/1) into ANF coefficients indexed by mask.
    """
    a = f_vals[:]
    for i in range(n):
        step = 1 << i
        for mask in range(1 << n):
            if mask & step:
                a[mask] ^= a[mask ^ step]
    return a


def truth_bit_from_sbox(S, n, m, out_bit_idx):
    """
    Extract one output bit truth table from the S-box.

    out_bit_idx ranges from 0 to m-1, where 0 is the MSB of the output.
    """
    return [(S[x] >> (m - 1 - out_bit_idx)) & 1 for x in range(1 << n)]


def anf_masks_from_sbox(S):
    """
    Return:
      n, m, masks_per_bit[k]

    where masks_per_bit[k] is the list of ANF monomial masks
    whose coefficient is 1 for output bit k.
    """
    n, m = infer_nm(S)
    masks_per_bit = []

    for k in range(m):
        f = truth_bit_from_sbox(S, n, m, k)
        coeffs = mobius_anf(f, n)
        masks = [mask for mask, c in enumerate(coeffs) if c]
        masks_per_bit.append(masks)

    return n, m, masks_per_bit


# =========================================================
# 3) ANF formatting and printing
# =========================================================
def mask_to_monomial(mask, n, var_prefix="x"):
    """
    Convert one ANF mask into a monomial string.

    Convention:
      leftmost bit of the mask corresponds to x0
      rightmost bit corresponds to x(n-1)

    Examples:
      mask = 0       -> "1"
      mask = 0b10010 -> "x0x3"
    """
    if mask == 0:
        return "1"

    vars_in_term = []
    for i in range(n):
        if (mask >> (n - 1 - i)) & 1:
            vars_in_term.append(f"{var_prefix}{i}")

    return "".join(vars_in_term)


def anf_masks_to_string(masks, n, var_prefix="x", xor_symbol="⊕"):
    """
    Convert a list of ANF masks into a readable ANF expression.
    """
    if not masks:
        return "0"

    masks_sorted = sorted(masks, key=lambda mask: (popcount(mask), mask))
    return f" {xor_symbol} ".join(mask_to_monomial(mask, n, var_prefix) for mask in masks_sorted)


def print_anf_functions(anf_masks, n, m, label="", out_prefix="y", xor_symbol="⊕"):
    """Print all ANF coordinate functions."""
    title = f"=== ANF functions for {label} ===" if label else "=== ANF functions ==="
    print(f"\n{title}")
    print(f"(Output bits are printed in MSB -> LSB order: {out_prefix}0, {out_prefix}1, ..., {out_prefix}{m-1})")
    for k in range(m):
        expr = anf_masks_to_string(anf_masks[k], n, xor_symbol=xor_symbol)
        print(f"{out_prefix}{k} = {expr}")


# =========================================================
# 4) ANF statistics
# =========================================================
def anf_stats(anf_masks):
    """
    Compute useful ANF statistics:
      - total number of monomials
      - maximum algebraic degree
      - number of degree >= 3 monomials
      - degree histogram
      - per-bit number of monomials
      - per-bit maximum degree
    """
    deg_hist = defaultdict(int)
    per_bit_terms = []
    per_bit_maxdeg = []
    total_terms = 0
    max_deg = 0
    hi_deg_terms = 0

    for masks in anf_masks:
        total_terms += len(masks)
        per_bit_terms.append(len(masks))

        bit_max_deg = 0
        for mask in masks:
            d = popcount(mask)
            deg_hist[d] += 1
            bit_max_deg = max(bit_max_deg, d)
            max_deg = max(max_deg, d)
            if d >= 3:
                hi_deg_terms += 1

        per_bit_maxdeg.append(bit_max_deg)

    return {
        "total_terms": int(total_terms),
        "max_deg": int(max_deg),
        "hi_deg_terms": int(hi_deg_terms),
        "deg_hist": dict(sorted(deg_hist.items())),
        "per_bit_terms": per_bit_terms,
        "per_bit_maxdeg": per_bit_maxdeg,
    }


def required_clean_ancilla_from_degree(d_max):
    """
    For the AND-chain clean-ancilla construction,
    a monomial of degree d >= 3 needs (d - 2) clean ancilla qubits.
    """
    return max(0, d_max - 2)


# =========================================================
# 5) Logical resource estimation without transpilation
# =========================================================
def estimate_resources_no_ancilla(n, m, anf_masks):
    """
    Estimate logical resources for the direct no-ancilla implementation.

    Mapping:
      degree 0   -> X
      degree 1   -> CNOT
      degree 2   -> Toffoli
      degree >=3 -> one MCX-like gate of degree d
    """
    x_count = 0
    cnot_count = 0
    toffoli_count = 0
    mcx_like_count = 0
    mcx_ctrl_hist = defaultdict(int)
    total_ops = 0

    for masks in anf_masks:
        for mask in masks:
            d = popcount(mask)
            if d == 0:
                x_count += 1
            elif d == 1:
                cnot_count += 1
            elif d == 2:
                toffoli_count += 1
            else:
                mcx_like_count += 1
                mcx_ctrl_hist[d] += 1
            total_ops += 1

    return {
        "width": n + m,
        "ancilla": 0,
        "x": x_count,
        "cnot": cnot_count,
        "toffoli": toffoli_count,
        "mcx_like": mcx_like_count,
        "mcx_ctrl_hist": dict(sorted(mcx_ctrl_hist.items())),
        "total_ops": total_ops,
        "depth_seq": total_ops,
    }


def estimate_resources_clean_ancilla(n, m, anf_masks, work):
    """
    Estimate logical resources for the clean-ancilla compute-toggle-uncompute implementation.

    Mapping:
      degree 0   -> X
      degree 1   -> CNOT
      degree 2   -> Toffoli
      degree >=3 -> chain + toggle + uncompute

    For one monomial of degree d >= 3:
      required clean ancilla = d - 2
      Toffoli count          = 2d - 3
    """
    x_count = 0
    cnot_count = 0
    toffoli_count = 0
    total_ops = 0

    for masks in anf_masks:
        for mask in masks:
            d = popcount(mask)

            if d == 0:
                x_count += 1
                total_ops += 1
            elif d == 1:
                cnot_count += 1
                total_ops += 1
            elif d == 2:
                toffoli_count += 1
                total_ops += 1
            else:
                need = d - 2
                if work < need:
                    raise ValueError(f"Need >= {need} clean ancilla, but work={work}.")
                add_toffoli = 2 * d - 3
                toffoli_count += add_toffoli
                total_ops += add_toffoli

    return {
        "width": n + m + work,
        "ancilla": work,
        "x": x_count,
        "cnot": cnot_count,
        "toffoli": toffoli_count,
        "total_ops": total_ops,
        "depth_seq": total_ops,
    }


# =========================================================
# 6) Qiskit circuit builders
# =========================================================
def require_qiskit():
    """Raise an error if Qiskit is not installed."""
    if not HAS_QISKIT:
        raise ImportError(
            "Qiskit is not installed. Install it with: pip install qiskit"
        )


def _apply_mcx(qc, ctrls, tgt):
    """
    Apply a controlled-X family gate:
      0 controls -> X
      1 control  -> CX
      2 controls -> CCX
      >=3        -> generic MCX
    """
    d = len(ctrls)
    if d == 0:
        qc.x(tgt)
    elif d == 1:
        qc.cx(ctrls[0], tgt)
    elif d == 2:
        qc.ccx(ctrls[0], ctrls[1], tgt)
    else:
        qc.append(XGate().control(d), list(ctrls) + [tgt])


def circuit_no_ancilla(n, m, anf_masks, name="sbox_no_anc"):
    """
    Build |x>|0^m> -> |x>|S(x)> using direct monomial implementation.
    """
    require_qiskit()

    qc = QuantumCircuit(n + m, name=name)
    x = [qc.qubits[i] for i in range(n)]
    y = [qc.qubits[n + k] for k in range(m)]

    for k in range(m):
        for mask in anf_masks[k]:
            ctrls = [x[i] for i in range(n) if (mask >> (n - 1 - i)) & 1]
            _apply_mcx(qc, ctrls, y[k])

    return qc


def circuit_clean_ancilla(n, m, anf_masks, work, name="sbox_clean_anc"):
    """
    Build |x>|0^m>|0^work> -> |x>|S(x)>|0^work> using a clean-ancilla AND-chain.

    For a monomial of degree d:
      d = 0 -> X
      d = 1 -> CX
      d = 2 -> CCX
      d >=3 -> compute chain, toggle target, uncompute chain
    """
    require_qiskit()

    qc = QuantumCircuit(n + m + work, name=name)
    x = [qc.qubits[i] for i in range(n)]
    y = [qc.qubits[n + k] for k in range(m)]
    a = [qc.qubits[n + m + t] for t in range(work)]

    for k in range(m):
        for mask in anf_masks[k]:
            ctrls_idx = [i for i in range(n) if (mask >> (n - 1 - i)) & 1]
            d = len(ctrls_idx)

            if d == 0:
                qc.x(y[k])
                continue

            if d == 1:
                qc.cx(x[ctrls_idx[0]], y[k])
                continue

            if d == 2:
                qc.ccx(x[ctrls_idx[0]], x[ctrls_idx[1]], y[k])
                continue

            need = d - 2
            if work < need:
                raise ValueError(f"Need >= {need} clean ancilla, but work={work}.")

            c = [x[i] for i in ctrls_idx]

            qc.ccx(c[0], c[1], a[0])
            for j in range(2, d - 1):
                qc.ccx(a[j - 2], c[j], a[j - 1])

            qc.ccx(a[need - 1], c[d - 1], y[k])

            for j in reversed(range(2, d - 1)):
                qc.ccx(a[j - 2], c[j], a[j - 1])
            qc.ccx(c[0], c[1], a[0])

    return qc


# =========================================================
# 7) Qiskit metrics after transpilation
# =========================================================
def _filter_non_gates(ops):
    """Remove pseudo-operations such as barrier and measure."""
    drop = {"barrier", "measure", "snapshot", "delay"}
    return {g: int(v) for g, v in ops.items() if g not in drop}


def logical_counts(qc):
    """
    Count raw logical operations in the original circuit.
    """
    cnt = Counter()
    for inst in qc.data:
        op = getattr(inst, "operation", None)
        if op is None:
            op = inst[0]
        cnt[op.name] += 1

    mcx_like = 0
    for name, v in cnt.items():
        nm = name.lower()
        if ("mcx" in nm) or (nm.startswith("c") and nm.endswith("x") and nm not in ("cx", "ccx")):
            mcx_like += v

    return {
        "width_raw": qc.num_qubits,
        "depth_raw": qc.depth(),
        "x": int(cnt.get("x", 0)),
        "cx": int(cnt.get("cx", 0)),
        "ccx": int(cnt.get("ccx", 0)),
        "mcx_like": int(mcx_like),
        "total_ops": int(sum(cnt.values())),
    }


def layer_count_of_gate(circ, gate_names):
    """
    Count how many DAG layers contain at least one gate from gate_names.
    """
    dag = circuit_to_dag(circ)
    layers = 0
    for layer in dag.layers():
        ops = layer["graph"].op_nodes()
        if any(op.name in gate_names for op in ops):
            layers += 1
    return int(layers)


def compile_once(qc, basis, opt_level, seed):
    """
    Transpile one circuit once for a given basis and seed.
    """
    return transpile(
        qc,
        basis_gates=basis,
        optimization_level=opt_level,
        seed_transpiler=seed,
    )


def compiled_metrics_hw(qc, opt_level, seeds, basis_hw=("cx", "rz", "sx", "x", "h")):
    """
    Compile to a hardware-like basis and report min / median / max metrics.
    """
    depths, twoq_depths, cxs, oneqs, totals = [], [], [], [], []

    for sd in seeds:
        tqc = compile_once(qc, list(basis_hw), opt_level, sd)
        ops = _filter_non_gates(tqc.count_ops())

        depth = int(tqc.depth())
        cx = int(ops.get("cx", 0))
        oneq = int(sum(v for g, v in ops.items() if g != "cx"))
        total = int(sum(ops.values()))
        twoq_depth = layer_count_of_gate(tqc, {"cx"})

        depths.append(depth)
        cxs.append(cx)
        oneqs.append(oneq)
        totals.append(total)
        twoq_depths.append(twoq_depth)

    return {
        "basis": "HW",
        "width": qc.num_qubits,
        "depth": agg_min_med_max(depths),
        "twoq_depth": agg_min_med_max(twoq_depths),
        "cx": agg_min_med_max(cxs),
        "oneq_total": agg_min_med_max(oneqs),
        "gates_total": agg_min_med_max(totals),
        "seeds": list(seeds),
    }


def compiled_metrics_nct(qc, opt_level, seeds, basis_nct=("x", "cx", "ccx")):
    """
    Try to compile to the NCT basis and report min / median / max metrics.

    If transpilation fails, return unavailable.
    """
    depths, cxs, ccxs, totals, toff_depths = [], [], [], [], []
    ok = True

    for sd in seeds:
        try:
            tqc = compile_once(qc, list(basis_nct), opt_level, sd)
        except Exception:
            ok = False
            break

        ops = _filter_non_gates(tqc.count_ops())
        depth = int(tqc.depth())
        cx = int(ops.get("cx", 0))
        ccx = int(ops.get("ccx", 0))
        total = int(sum(ops.values()))
        td = layer_count_of_gate(tqc, {"ccx"})

        depths.append(depth)
        cxs.append(cx)
        ccxs.append(ccx)
        totals.append(total)
        toff_depths.append(td)

    if not ok:
        return {"basis": "NCT", "available": False}

    return {
        "basis": "NCT",
        "available": True,
        "width": qc.num_qubits,
        "depth": agg_min_med_max(depths),
        "cx": agg_min_med_max(cxs),
        "ccx": agg_min_med_max(ccxs),
        "toffoli_depth": agg_min_med_max(toff_depths),
        "gates_total": agg_min_med_max(totals),
        "seeds": list(seeds),
    }


def hw_gate_breakdown_per_seed(qc, seeds, opt_level=2, basis_hw=("cx", "rz", "sx", "x", "h")):
    """
    Return detailed per-seed hardware-basis gate counts after transpilation.
    """
    rows = []

    for sd in seeds:
        tqc = transpile(
            qc,
            basis_gates=list(basis_hw),
            optimization_level=int(opt_level),
            seed_transpiler=int(sd),
        )
        ops = _filter_non_gates(tqc.count_ops())

        row = {
            "seed": int(sd),
            "depth": int(tqc.depth()),
            "twoq_depth": int(layer_count_of_gate(tqc, {"cx"})),
        }

        for g in basis_hw:
            row[g] = int(ops.get(g, 0))

        other = {g: int(v) for g, v in ops.items() if g not in set(basis_hw)}
        row["other"] = other
        row["total"] = int(sum(ops.values()))
        rows.append(row)

    return rows


def print_hw_gate_breakdown(qc, title, seeds, opt_level=2, basis_hw=("cx", "rz", "sx", "x", "h")):
    """
    Print detailed per-seed hardware-basis gate counts after transpilation.
    """
    rows = hw_gate_breakdown_per_seed(qc, seeds, opt_level=opt_level, basis_hw=basis_hw)

    print(f"\n=== HW gate breakdown ({title}) | basis={list(basis_hw)} | opt_level={opt_level} ===")
    for r in rows:
        other_str = f" | other={r['other']}" if r["other"] else ""
        print(
            f" seed={r['seed']}: depth={r['depth']} twoq_depth={r['twoq_depth']} | "
            f"cx={r['cx']} rz={r['rz']} sx={r['sx']} x={r['x']} h={r['h']} | total={r['total']}"
            f"{other_str}"
        )

    print("  min/med/max:")
    for key in ["depth", "twoq_depth", "cx", "rz", "sx", "x", "h", "total"]:
        xs = [r[key] for r in rows]
        d = agg_min_med_max(xs)
        print(f"   - {key:>10}: min={d['min']:.0f}, med={d['med']:.0f}, max={d['max']:.0f}")


# =========================================================
# 8) Evaluate one S-box
# =========================================================
def evaluate_sbox(S, label, seeds=None, opt_level_hw=2, opt_level_nct=1):
    """
    Evaluate one S-box:
      - infer dimensions
      - compute ANF
      - compute ANF statistics
      - estimate logical resources
      - if Qiskit is available, build circuits and report post-transpile metrics
    """
    if seeds is None:
        seeds = [1, 2, 0, 4, 3]

    n, m, anf_masks = anf_masks_from_sbox(S)
    astats = anf_stats(anf_masks)
    work = required_clean_ancilla_from_degree(astats["max_deg"])

    res = {
        "label": label,
        "n": n,
        "m": m,
        "bijective": is_bijective(S),
        "anf_masks": anf_masks,
        "anf": astats,
        "clean_work": work,
        "qiskit_available": HAS_QISKIT,
        "no_ancilla": {
            "estimate": estimate_resources_no_ancilla(n, m, anf_masks)
        },
        "clean_ancilla": {
            "estimate": estimate_resources_clean_ancilla(n, m, anf_masks, work)
        },
    }

    if HAS_QISKIT:
        qc_no = circuit_no_ancilla(n, m, anf_masks, name=f"{label}_noanc")
        qc_ca = circuit_clean_ancilla(n, m, anf_masks, work=work, name=f"{label}_cleananc")

        res["no_ancilla"]["raw"] = logical_counts(qc_no)
        res["clean_ancilla"]["raw"] = logical_counts(qc_ca)

        res["no_ancilla"]["hw"] = compiled_metrics_hw(qc_no, opt_level_hw, seeds)
        res["clean_ancilla"]["hw"] = compiled_metrics_hw(qc_ca, opt_level_hw, seeds)

        res["no_ancilla"]["nct"] = compiled_metrics_nct(qc_no, opt_level_nct, seeds)
        res["clean_ancilla"]["nct"] = compiled_metrics_nct(qc_ca, opt_level_nct, seeds)

        res["no_ancilla"]["qc"] = qc_no
        res["clean_ancilla"]["qc"] = qc_ca

    return res


# =========================================================
# 9) Summary printing
# =========================================================
def _fmt_mmm(d):
    """Format min / median / max dictionaries."""
    if d["min"] is None:
        return "N/A"
    return f"min={d['min']:.0f}, med={d['med']:.0f}, max={d['max']:.0f}"


def print_summary(res):
    """
    Print ANF statistics, logical estimates,
    and Qiskit post-transpile summaries if available.
    """
    print(f"\n=== {res['label']} | n={res['n']} m={res['m']} | bijective={res['bijective']} ===")

    a = res["anf"]
    print(f"ANF total_terms      = {a['total_terms']}")
    print(f"ANF max_deg          = {a['max_deg']}")
    print(f"ANF hi_deg_terms>=3  = {a['hi_deg_terms']}")
    print(f"ANF per-bit terms    = {a['per_bit_terms']}")
    print(f"ANF per-bit max deg  = {a['per_bit_maxdeg']}")
    print(f"ANF degree hist      = {a['deg_hist']}")
    print(f"clean ancilla needed = {res['clean_work']}")

    for mode in ["no_ancilla", "clean_ancilla"]:
        est = res[mode]["estimate"]
        print(f"\n[{mode}] logical estimate")
        print(f" width         = {est['width']}")
        print(f" ancilla       = {est['ancilla']}")
        print(f" X             = {est['x']}")
        print(f" CNOT          = {est['cnot']}")
        print(f" Toffoli       = {est['toffoli']}")
        if "mcx_like" in est:
            print(f" MCX_like      = {est['mcx_like']}")
            print(f" MCX ctrl hist = {est['mcx_ctrl_hist']}")
        print(f" total_ops     = {est['total_ops']}")
        print(f" depth_seq     = {est['depth_seq']}")

        if res["qiskit_available"]:
            raw = res[mode]["raw"]
            hw = res[mode]["hw"]
            nct = res[mode]["nct"]

            print(f"\n[{mode}] raw Qiskit circuit")
            print(
                f" width={raw['width_raw']} depth={raw['depth_raw']} | "
                f"X={raw['x']} CX={raw['cx']} CCX={raw['ccx']} "
                f"MCX~={raw['mcx_like']} total_ops={raw['total_ops']}"
            )

            print(f"[{mode}] post-transpile HW summary")
            print(
                f" width={hw['width']} | "
                f"depth({_fmt_mmm(hw['depth'])}) | "
                f"twoq_depth({_fmt_mmm(hw['twoq_depth'])}) | "
                f"CX({_fmt_mmm(hw['cx'])}) | "
                f"1q_total({_fmt_mmm(hw['oneq_total'])}) | "
                f"gates_total({_fmt_mmm(hw['gates_total'])})"
            )

            print(f"[{mode}] post-transpile NCT summary")
            if nct.get("available", False):
                print(
                    f" width={nct['width']} | "
                    f"depth({_fmt_mmm(nct['depth'])}) | "
                    f"CCX(Toffoli)({_fmt_mmm(nct['ccx'])}) | "
                    f"Toffoli-depth({_fmt_mmm(nct['toffoli_depth'])}) | "
                    f"CX({_fmt_mmm(nct['cx'])}) | "
                    f"gates_total({_fmt_mmm(nct['gates_total'])})"
                )
            else:
                print(" unavailable (cannot transpile fully to {x, cx, ccx} in this setup)")
        else:
            print("\nQiskit summary unavailable because Qiskit is not installed.")


# =========================================================
# 10) Main
# =========================================================
if __name__ == "__main__":
    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------
    SEEDS = [1, 2, 0, 4, 3]
    OPT_LEVEL_HW = 2
    OPT_LEVEL_NCT = 1
    PRINT_PER_SEED_BREAKDOWN = True

    # -----------------------------------------------------
    # Put your S-boxes here
    # -----------------------------------------------------
    sboxes = [
        ("Proposed Sbox", [1,5,9,14,0,4,10,13,3,15,11,2,6,8,12,7   ]),
    ]

    # -----------------------------------------------------
    # Evaluation loop
    # -----------------------------------------------------
    for label, S in sboxes:
        res = evaluate_sbox(
            S,
            label=label,
            seeds=SEEDS,
            opt_level_hw=OPT_LEVEL_HW,
            opt_level_nct=OPT_LEVEL_NCT
        )

        print_anf_functions(
            res["anf_masks"],
            res["n"],
            res["m"],
            label=label,
            out_prefix="y",
            xor_symbol="⊕"
        )

        print_summary(res)

        if HAS_QISKIT and PRINT_PER_SEED_BREAKDOWN:
            print_hw_gate_breakdown(
                res["no_ancilla"]["qc"],
                title=f"{label} no_ancilla",
                seeds=SEEDS,
                opt_level=OPT_LEVEL_HW
            )

            print_hw_gate_breakdown(
                res["clean_ancilla"]["qc"],
                title=f"{label} clean_ancilla",
                seeds=SEEDS,
                opt_level=OPT_LEVEL_HW
            )