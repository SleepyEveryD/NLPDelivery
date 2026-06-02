"""Type-specific DETERMINISTIC maths solvers -- the "calculator pointed at the right question" idea.

The lesson from the reverted general calculator (Q6767): letting the MODEL'S (often wrong) set-up drive
an override is fatal. So here WE own the solving logic per question-TYPE -- correct by construction -- and
each solver is silent (returns None) on anything outside its type. A solver fires ONLY when it (a) detects
its type with high precision, (b) parses the inputs cleanly, and (c) maps its result to EXACTLY ONE option.
Any miss -> None -> the LLM handles the question untouched. So the suite can only ADD correct answers on the
computational subset; it can never override the LLM on the knowledge/abstract subset (those types are simply
not claimed). See [[maths-live-routing-stack]].

`solve_maths(question) -> (letter, evidence) | None` is the single entry point.
"""
from __future__ import annotations

import ast
import datetime
import itertools
import math
import re
from fractions import Fraction

from schemas import Question


# ---------------------------------------------------------------------------
# Expression cleanup + Fraction-exact safe eval (no eval(), AST-walk only).
# ---------------------------------------------------------------------------
def _clean_expr(s: str) -> str:
    """LaTeX / formatting -> plain Python arithmetic. Best-effort, lossless for the forms we solve."""
    s = s.strip()
    s = s.replace("$", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*")
    s = s.replace("\\div", "/").replace("÷", "/")
    # \frac{a}{b} / \dfrac / \cfrac -> (a)/(b); loop a few times for limited nesting.
    for _ in range(4):
        new = re.sub(r"\\[dct]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
        if new == s:
            break
        s = new
    s = s.replace("^", "**")
    s = re.sub(r"\{(\d+)\}", r"\1", s)   # bare ^{n} braces left after ^-> **
    s = s.replace("{", "(").replace("}", ")")
    return s


_ALLOWED = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b, ast.FloorDiv: lambda a, b: a // b,
}


def _eval_node(node) -> Fraction:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("non-number")
        return Fraction(str(node.value))
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            base = _eval_node(node.left)
            exp = _eval_node(node.right)
            if exp.denominator != 1:
                raise ValueError("non-integer power")
            return base ** int(exp)
        op = _ALLOWED.get(type(node.op))
        if op is None:
            raise ValueError("op")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise ValueError("unary")
    raise ValueError("forbidden")


def _safe_eval(expr: str) -> Fraction | None:
    """A purely-numeric arithmetic expression -> exact Fraction, or None if it isn't one."""
    expr = _clean_expr(expr)
    if not expr or re.search(r"[A-Za-z\\]", expr):   # any letter/backslash left -> not pure numeric.
        return None
    try:
        return _eval_node(ast.parse(expr, mode="eval").body)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Option-matching helpers -- the gate that makes a solver safe (exactly-one match or abstain).
# ---------------------------------------------------------------------------
def _option_value(text: str) -> Fraction | None:
    """An option's text -> its numeric value (handles fractions, %, commas, 2^11-1), or None."""
    t = text.strip()
    t = re.sub(r"\b(inches|inch|points?|dollars?|degrees?|cm|m|kg|%)\b", "", t, flags=re.I)
    t = t.replace(",", "").replace("$", "").strip()
    return _safe_eval(t)


def _match_value(options: dict[str, str], target: Fraction, *, tol: Fraction | None = None) -> str | None:
    """The UNIQUE option letter whose numeric value equals target -- else None (0 or >1 matches)."""
    hits = []
    for k, v in options.items():
        ov = _option_value(v)
        if ov is None:
            continue
        if ov == target or (tol is not None and abs(ov - target) <= tol):
            hits.append(k)
    return hits[0] if len(hits) == 1 else None


def _int_set(text: str) -> set[int] | None:
    """An option like '0,4' or '0, 1, 2' -> {0,4} / {0,1,2}; None if it isn't a clean int list."""
    t = re.sub(r"[^\d,\-\s]", "", text.strip())
    parts = [p.strip() for p in re.split(r"[,\s]+", t) if p.strip() not in ("", "-")]
    try:
        return {int(p) for p in parts}
    except ValueError:
        return None


def _match_set(options: dict[str, str], target: set) -> str | None:
    hits = [k for k, v in options.items() if _int_set(v) == target]
    return hits[0] if len(hits) == 1 else None


def _moduli(text: str) -> list[int]:
    """Every Z_n subscript in the text -> [n, ...] (Z_4 x Z_12 -> [4, 12])."""
    return [int(m) for m in re.findall(r"\bZ_?\{?(\d+)\}?", text)]


# ---------------------------------------------------------------------------
# The solvers. Each returns (letter, evidence) or None (abstain).
# ---------------------------------------------------------------------------
def _solve_finite_field_roots(q: Question) -> tuple[str, str] | None:
    """'Find all zeros of <poly> in Z_n' -> evaluate the poly at every element of Z_n."""
    t = q.text
    if not re.search(r"\b(zeros?|roots?)\b", t, re.I):
        return None
    mods = _moduli(t)
    if not mods:
        return None
    n = mods[0]
    if not (2 <= n <= 97):
        return None
    # The polynomial: the run of x-terms (grab the longest such substring).
    cands = re.findall(r"[0-9x\^\+\-\*\s\{\}]*x[0-9x\^\+\-\*\s\{\}]*", _clean_expr(t).replace("**", "^"))
    coeffs = None
    for c in sorted(cands, key=len, reverse=True):
        coeffs = _parse_poly(c)
        if coeffs:
            break
    if not coeffs:
        return None
    roots = {c for c in range(n) if sum(co * (c ** p) for p, co in coeffs.items()) % n == 0}
    letter = _match_set(q.options, roots)
    if letter:
        return letter, f"finite-field roots in Z_{n}: {sorted(roots)}"
    return None


def _parse_poly(s: str) -> dict[int, int] | None:
    """A polynomial-in-x string -> {power: int coeff}, or None if any term won't parse."""
    s = _clean_expr(s).replace(" ", "")
    s = re.sub(r"in.*$", "", s)
    if "x" not in s:
        return None
    s = s.replace("-", "+-")
    coeffs: dict[int, int] = {}
    for term in (p for p in s.split("+") if p):
        m = re.fullmatch(r"(-?\d*)\*?x(?:\*\*(\d+))?", term)
        if m:
            c = m.group(1)
            c = -1 if c == "-" else (1 if c == "" else int(c))
            p = int(m.group(2)) if m.group(2) else 1
        elif re.fullmatch(r"-?\d+", term):
            c, p = int(term), 0
        else:
            return None      # an unparseable term -> abstain on the whole question.
        coeffs[p] = coeffs.get(p, 0) + c
    return coeffs or None


def _solve_ring_characteristic(q: Question) -> tuple[str, str] | None:
    """'characteristic of the ring Z_m x Z_n' -> lcm(m, n) (single Z_n -> n)."""
    if "characteristic" not in q.text.lower():
        return None
    mods = _moduli(q.text)
    if not mods:
        return None
    char = mods[0]
    for m in mods[1:]:
        char = char * m // math.gcd(char, m)
    letter = _match_value(q.options, Fraction(char))
    if letter:
        return letter, f"ring characteristic = lcm{tuple(mods)} = {char}"
    return None


def _solve_gcd(q: Question) -> tuple[str, str] | None:
    """'gcd of m^a-1 and m^b-1' = m^gcd(a,b)-1; or gcd of two plain integers."""
    if not re.search(r"greatest common divisor|\bgcd\b", q.text, re.I):
        return None
    powers = re.findall(r"(\d+)\s*\^\s*\{?(\d+)\}?\s*-\s*1", q.text)
    if len(powers) >= 2 and powers[0][0] == powers[1][0]:
        base = int(powers[0][0])
        a, b = int(powers[0][1]), int(powers[1][1])
        val = Fraction(base ** math.gcd(a, b) - 1)
        # options may be numeric OR symbolic ('2^11 - 1'); _option_value handles both via _safe_eval.
        letter = _match_value(q.options, val)
        if letter:
            return letter, f"gcd({base}^{a}-1, {base}^{b}-1) = {base}^{math.gcd(a,b)}-1 = {val}"
        return None
    ints = [int(x.replace(",", "")) for x in re.findall(r"\bof\s+([\d,]+)\s+and\s+([\d,]+)", q.text)[0]] \
        if re.search(r"\bof\s+[\d,]+\s+and\s+[\d,]+", q.text) else []
    if len(ints) == 2:
        letter = _match_value(q.options, Fraction(math.gcd(ints[0], ints[1])))
        if letter:
            return letter, f"gcd{tuple(ints)} = {math.gcd(*ints)}"
    return None


def _solve_sum_product(q: Question) -> tuple[str, str] | None:
    """'Two numbers ... sum is S ... product is P' -> solve t^2 - S t + P = 0."""
    t = q.text
    if not re.search(r"\btwo numbers\b", t, re.I):
        return None
    s_m = re.search(r"(?:added together|sum|add up)[^\d]*(\d+)", t, re.I)
    p_m = re.search(r"product[^\d]*(\d+)", t, re.I)
    if not (s_m and p_m):
        return None
    S, P = int(s_m.group(1)), int(p_m.group(1))
    disc = S * S - 4 * P
    if disc < 0:
        return None
    r = math.isqrt(disc)
    if r * r != disc or (S + r) % 2:
        return None
    pair = {(S + r) // 2, (S - r) // 2}
    letter = _match_set(q.options, pair)
    if letter:
        return letter, f"two numbers with sum {S}, product {P}: {sorted(pair)}"
    return None


def _solve_reflection_yx(q: Question) -> tuple[str, str] | None:
    """'<polygon> reflected across y = x' -> reflect every (a,b) to (b,a); match an option point."""
    t = q.text
    if not (re.search(r"reflect", t, re.I) and re.search(r"y\s*=\s*x", t)):
        return None
    pt = r"\(\s*(-?\s*\d+)\s*,\s*(-?\s*\d+)\s*\)"   # allow a space after the minus: '(- 2, - 4)'.
    def _i(s): return int(s.replace(" ", ""))
    verts = re.findall(pt, t)
    if len(verts) < 2:
        return None
    reflected = {(_i(b), _i(a)) for a, b in verts}
    hits = []
    for k, v in q.options.items():
        m = re.search(pt, v)
        if m and (_i(m.group(1)), _i(m.group(2))) in reflected:
            hits.append(k)
    if len(hits) == 1:
        return hits[0], f"reflection across y=x; P' contains {sorted(reflected)}"
    return None


def _solve_triangle_sides(q: Question) -> tuple[str, str] | None:
    """'Which could NOT be the sides of an (isosceles) triangle?' -> test each option triple."""
    t = q.text
    if not re.search(r"sides?\b.*triangle", t, re.I):
        return None
    negated = bool(re.search(r"\bnot\b|cannot|could\s+not", t, re.I))
    iso = "isosceles" in t.lower()

    def ok(sides: list[int]) -> bool:
        a, b, c = sorted(sides)
        valid = a + b > c          # strict triangle inequality (degenerate -> not a triangle).
        if iso:
            return valid and len({a, b, c}) <= 2
        return valid

    parsed = {}
    for k, v in q.options.items():
        nums = re.findall(r"\d+(?:\.\d+)?", v)
        if len(nums) != 3:
            return None            # an option that isn't a clean triple -> abstain.
        parsed[k] = [float(x) for x in nums]
    failing = [k for k, s in parsed.items() if not ok(s)]
    passing = [k for k, s in parsed.items() if ok(s)]
    want = failing if negated else passing
    if len(want) == 1:
        return want[0], f"triangle-inequality{'/isosceles' if iso else ''} test -> option {want[0]}"
    return None


def _solve_percentage_increase(q: Question) -> tuple[str, str] | None:
    """'<N> ... increases by P% ... (to the nearest whole)' -> N*(1+P/100), rounded if asked."""
    t = q.text
    m = re.search(r"increase[sd]?\s+by\s+(\d+(?:\.\d+)?)\s*%", t, re.I)
    if not m:
        return None
    p = Fraction(m.group(1))
    # the base = the number immediately before 'increase' (nearest preceding standalone number).
    pre = t[: m.start()]
    nums = re.findall(r"(\d+(?:\.\d+)?)", pre)
    if not nums:
        return None
    base = Fraction(nums[-1])
    val = base * (1 + p / 100)
    if re.search(r"nearest\s+whole", t, re.I):
        val = Fraction(round(val))
    letter = _match_value(q.options, val)
    if letter:
        return letter, f"{base} increased by {p}% = {float(val):g}"
    return None


def _solve_arith_expression(q: Question) -> tuple[str, str] | None:
    """A self-contained numeric expression (after Find/Simplify/Evaluate/compute) -> its exact value."""
    t = q.text
    m = re.search(r"\b(?:find|simplify|evaluate|compute|value of)\b(.+)", t, re.I | re.S)
    if not m:
        return None
    expr = m.group(1).strip().rstrip(".?")
    val = _safe_eval(expr)
    if val is None:
        return None
    letter = _match_value(q.options, val)
    if letter:
        return letter, f"evaluated expression = {val}"
    return None


def _solve_common_divisor_count(q: Question) -> tuple[str, str] | None:
    """'How many positive integers are factors of A and (also) factors of B' -> d(gcd(A,B))."""
    t = q.text
    if not (re.search(r"how many", t, re.I) and re.search(r"\b(?:factors?|divisors?)\b", t, re.I)):
        return None
    nums = None
    for pat in (
        r"common\s+(?:factors?|divisors?)\s+of\s+(\d+)\s+and\s+(\d+)",
        r"(?:factors?|divisors?)\s+of\s+(\d+)\b[\s\S]*?(?:factors?|divisors?)\s+of\s+(\d+)",
        r"(?:factors?|divisors?)\s+of\s+(\d+)\s+and\s+(?:also\s+)?(?:of\s+)?(\d+)",
    ):
        m = re.search(pat, t, re.I)
        if m:
            nums = (int(m.group(1)), int(m.group(2)))
            break
    if not nums:
        return None
    g = math.gcd(*nums)
    cnt = sum(1 for k in range(1, g + 1) if g % k == 0)
    letter = _match_value(q.options, Fraction(cnt))
    if letter:
        return letter, f"common divisors of {nums} = d(gcd={g}) = {cnt}"
    return None


def _solve_subspace_intersection_dim(q: Question) -> tuple[str, str] | None:
    """'a-dim V, b-dim W in n-dim X; which (cannot) be dim(V∩W)' -> range [max(0,a+b-n), min(a,b)]."""
    t = q.text
    if not (re.search(r"subspace", t, re.I) and re.search(r"inters|∩", t, re.I)):
        return None
    dims = [int(x) for x in re.findall(r"(\d+)\s*-?\s*dimensional", t, re.I)]
    if len(dims) < 2:
        return None
    n = max(dims)
    others = [d for d in dims if d < n]
    if not others:
        return None
    a = others[0]
    b = others[1] if len(others) > 1 else others[0]
    if a > n or b > n:
        return None
    lo, hi = max(0, a + b - n), min(a, b)
    valid = set(range(lo, hi + 1))
    opts = {}
    for k, v in q.options.items():
        m = re.fullmatch(r"\s*(\d+)\s*", v)
        if not m:
            return None
        opts[k] = int(m.group(1))
    negated = bool(re.search(r"\bcannot\b|\bnot\b|impossible", t, re.I))
    if negated:
        cands = [k for k, val in opts.items() if val not in valid]
    else:
        cands = [k for k, val in opts.items() if val in valid]
    if len(cands) == 1:
        side = "outside" if negated else "inside"
        return cands[0], f"dim(V∩W) in [{lo},{hi}] (dims {a},{b} in {n}); {side} -> {cands[0]}"
    return None


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], 1)}
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def _solve_operator_placement(q: Question) -> tuple[str, str] | None:
    """'Place +, ×, − (each once) in N _ N _ N _ N to get the highest/lowest' -> enumerate placements."""
    t = q.text
    if not (re.search(r"(?:put|place|insert)\b[\s\S]{0,40}symbols?|each symbol", t, re.I)
            and re.search(r"\b(highest|largest|greatest|maximum|lowest|smallest|least|minimum)\b", t, re.I)):
        return None
    ops = [o for w, o in (("plus", "+"), ("times", "*"), ("minus", "-"), ("divided", "/"))
           if re.search(r"\b" + w, t, re.I)]
    if len(ops) < 2:
        return None
    region = t
    m = re.search(r"\\\[(.+?)\\\]", t, re.S)        # the \[...\] display block, if present.
    if m:
        region = m.group(1)
    region = re.sub(r"\\hphantom\{[^}]*\}", "", region)   # \hphantom{8} hides a STRAY digit -- drop it first.
    region = re.sub(r"\\[a-zA-Z]+", "", region)           # then any other LaTeX command (\underline, ...).
    nums = [int(x) for x in re.findall(r"-?\d+", region)]
    if len(nums) != len(ops) + 1:
        return None
    maximize = bool(re.search(r"\b(highest|largest|greatest|maximum)\b", t, re.I))
    vals = []
    for perm in set(itertools.permutations(ops)):
        expr = str(nums[0]) + "".join(o + str(n) for o, n in zip(perm, nums[1:]))
        v = _safe_eval(expr)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    target = max(vals) if maximize else min(vals)
    letter = _match_value(q.options, target)
    if letter:
        return letter, f"{'max' if maximize else 'min'} of {ops} placed in {nums} = {target}"
    return None


def _solve_weekday(q: Question) -> tuple[str, str] | None:
    """'On what day of the week will <Month> <Year> begin?' -> real calendar day-of-week of the 1st."""
    t = q.text
    if not re.search(r"day of (?:the )?week", t, re.I):
        return None
    if not re.search(r"\bbegin|\bstart|first day", t, re.I):
        return None
    # If the stem GIVES an anchor weekday ('... is a Monday'), the answer is RELATIVE -- abstain.
    if re.search(r"\b(" + "|".join(_WEEKDAYS) + r")\b", t, re.I):
        return None
    ym = re.search(r"\b((?:19|20)\d{2})\b", t)
    if not ym:
        return None
    year = int(ym.group(1))
    month = None
    for mo in re.finditer(r"\b(" + "|".join(_MONTHS) + r")\b", t, re.I):
        month = _MONTHS[mo.group(1).lower()]   # the LAST month named is the one asked about.
    if month is None:
        return None
    try:
        wd = datetime.date(year, month, 1).strftime("%A")
    except Exception:
        return None
    hits = [k for k, v in q.options.items() if wd.lower() in v.lower()]
    if len(hits) == 1:
        return hits[0], f"1/{month}/{year} is a {wd}"
    return None


def _solve_parallelogram_angle(q: Question) -> tuple[str, str] | None:
    """'In parallelogram ABCD, angle B = X°, find angle C' -> opposite equal, adjacent supplementary."""
    t = q.text
    if "parallelogram" not in t.lower():
        return None
    name = re.search(r"parallelogram\s+\$?([A-Z]{4})\$?", t)
    given = re.search(r"angle\s+\$?([A-Z])\$?\s+(?:measures?|is|equals?|=)\s*\$?(\d+)", t, re.I)
    asked = re.search(r"(?:measure of|degrees? in[\s\S]{0,30}?)\s*angle\s+\$?([A-Z])\$?", t, re.I)
    if not (name and given and asked):
        return None
    verts, (gl, gv), al = name.group(1), (given.group(1), int(given.group(2))), asked.group(1)
    if gl not in verts or al not in verts:
        return None
    diff = abs(verts.index(gl) - verts.index(al)) % 4
    val = gv if diff in (0, 2) else 180 - gv      # opposite (0/2) equal; adjacent (1/3) supplementary.
    letter = _match_value(q.options, Fraction(val))
    if letter:
        return letter, f"parallelogram {verts}: angle {al} = {val} (angle {gl}={gv}, {'opposite' if diff in (0,2) else 'adjacent'})"
    return None


_SOLVERS = (
    _solve_finite_field_roots,
    _solve_ring_characteristic,
    _solve_gcd,
    _solve_common_divisor_count,
    _solve_subspace_intersection_dim,
    _solve_sum_product,
    _solve_reflection_yx,
    _solve_triangle_sides,
    _solve_percentage_increase,
    _solve_operator_placement,
    _solve_weekday,
    _solve_parallelogram_angle,
    _solve_arith_expression,
)


_DASHES = {ord(c): "-" for c in "‐‑‒–—―−"}  # ‐‑‒–—―−  -> ASCII '-'.


def _norm(s: str) -> str:
    """Unicode dashes/minus -> ASCII hyphen (the live data writes '(– 2, – 4)', not '-2'); NBSP -> space."""
    return (s or "").translate(_DASHES).replace(" ", " ")


def solve_maths(question: Question) -> tuple[str, str] | None:
    """Try every type-specific solver; the first confident, single-option hit wins. None = defer to LLM.

    Crash-safe: any solver that raises is skipped (it just abstains). MCQ-only -- no options, no solve.
    """
    if not question.options:
        return None
    # Normalise unicode dashes/minus in BOTH the stem and the options before any parser sees them.
    q = Question(
        qid=question.qid,
        text=_norm(question.text),
        options={k: _norm(v) for k, v in question.options.items()},
        qtype=question.qtype,
    )
    for fn in _SOLVERS:
        try:
            res = fn(q)
        except Exception:
            res = None
        if res is not None:
            return res
    return None
