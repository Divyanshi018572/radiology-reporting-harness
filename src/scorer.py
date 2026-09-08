"""Stage 5 local RES scorer: normalization, weighted edit, field-aware F, RES."""
import re
import unicodedata

NEGATION = {"no", "not", "without", "absent", "none", "negative", "neither",
            "nor", "never"}
LATERALITY = {"right", "left", "bilateral", "unilateral", "ipsilateral",
              "contralateral", "rt", "lt"}
SEVERITY = {"trace", "tiny", "small", "mild", "moderate", "marked", "large",
            "severe", "extensive", "minimal", "subtle", "significant", "gross"}
ACUITY = {"acute", "subacute", "chronic", "stable", "interval", "new",
          "resolved", "resolving", "worsening", "improving", "unchanged",
          "persistent"}
FUNCTION = {"the", "a", "an", "and", "or", "of", "with", "in", "on", "to",
            "for", "is", "are", "was", "were", "be", "been", "being", "by",
            "as", "at", "from", "into", "that", "this", "these", "those",
            "it", "its", "there", "their"}
UNITS = {"mm", "cm", "ml", "cc", "mg", "g", "kg", "mh", "hounsfield", "hu"}
NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
UNIT_MAP = {"millimeter": "mm", "millimeters": "mm", "millimetre": "mm",
            "millimetres": "mm", "centimeter": "cm", "centimeters": "cm",
            "centimetre": "cm", "centimetres": "cm", "cms": "cm", "mms": "mm"}

FIELD_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 /&\-']+):\s*(.*)$")


def norm_label(label: str) -> str:
    """Normalize field label for matching, return uppercased key."""
    return label.replace(":", "").strip().upper()


def normalize_text(text: str) -> list[str]:
    """Lowercase/normalize raw text into scored tokens, return token list."""
    s = unicodedata.normalize("NFKC", text).lower()
    signed = re.findall(r"(?<!\w)[+-]\d+(?:\.\d+)?", s)

    def _repl(m: re.Match) -> str:
        return f" qqsgn{_repl.i}qq "
    _repl.i = 0
    s = re.sub(r"(?<!\w)[+-]\d+(?:\.\d+)?", _repl, s)
    lines = [re.sub(r"^\s*(\d+[.\)]|[-*•])\s+", "", ln) for ln in s.splitlines()]
    s = "\n".join(lines)
    s = re.sub(r"(?<=[a-z])-(?=[a-z])", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    for i, orig in enumerate(signed):
        s = s.replace(f"qqsgn{i}qq", f" {orig.strip()} ")
    lines = [re.sub(r"^\s*\d+\s+", "", ln) for ln in s.splitlines()]
    s = "\n".join(lines)
    s = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", s)
    toks = []
    for t in s.split():
        toks.append(UNIT_MAP.get(t, t))
    return [t for t in toks if t]


def token_weight(tok: str) -> float:
    """Map single token to its RES weight, return 4.0/2.0/0.25."""
    if tok in NEGATION or tok in LATERALITY or tok in SEVERITY:
        return 4.0
    if tok in ACUITY or tok in UNITS or NUM_RE.match(tok):
        return 4.0
    if tok in FUNCTION:
        return 0.25
    return 2.0


def weighted_edit(ref: list[str], sub: list[str]) -> float:
    """Weighted word Levenshtein normalized by max weight, capped at 1."""
    rw = [token_weight(t) for t in ref]
    sw = [token_weight(t) for t in sub]
    n, m = len(ref), len(sub)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + rw[i - 1]
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + sw[j - 1]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == sub[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_c = max(rw[i - 1], sw[j - 1])
                dp[i][j] = min(dp[i - 1][j] + rw[i - 1],
                               dp[i][j - 1] + sw[j - 1],
                               dp[i - 1][j - 1] + sub_c)
    denom = max(sum(rw), sum(sw), 1e-9)
    return min(dp[n][m] / denom, 1.0)


def parse_report(text: str) -> tuple[dict[str, str], str]:
    """Split report into ordered {ORIGINAL_LABEL: text} + impression string."""
    parts = re.split(r"impression\s*:", text, flags=re.I, maxsplit=1)
    find_block = re.split(r"findings\s*:", parts[0], flags=re.I,
                           maxsplit=1)[-1]
    impression = parts[1].strip() if len(parts) > 1 else ""
    fields: dict[str, str] = {}
    current: str | None = None
    unlabelled: list[str] = []
    for line in find_block.splitlines():
        ml = FIELD_RE.match(line.strip())
        if ml:
            current = ml.group(1).strip()
            fields[current] = ml.group(2).strip()
        elif current is None:
            if line.strip():
                unlabelled.append(line.strip())
        elif line.strip():
            fields[current] += " " + line.strip()
    if unlabelled:
        fields["_UNLABELLED_"] = " ".join(unlabelled)
    return fields, impression


def res_case(reference: str, submitted: str, template: str) -> dict:
    """Score one case vs reference given template, return F/I/RES dict."""
    ref_f, ref_i = parse_report(reference)
    sub_f, sub_i = parse_report(submitted)
    tpl_f, _ = parse_report(template)
    tpl_norm = {norm_label(k): " ".join(normalize_text(v))
                for k, v in tpl_f.items()}
    sub_norm = {norm_label(k): k for k in sub_f}
    num, den, per_field = 0.0, 0.0, {}
    for label, ref_text in ref_f.items():
        nl = norm_label(label)
        if nl == "_UNLABELLED_":
            w = 3.0
            sub_text = sub_f.get(sub_norm.get(nl, ""), "")
        else:
            changed = tpl_norm.get(nl, None) != " ".join(
                normalize_text(ref_text))
            w = 3.0 if changed else 1.0
            sub_text = ""
            if nl in sub_norm:
                sub_text = sub_f[sub_norm[nl]]
        e = weighted_edit(normalize_text(ref_text), normalize_text(sub_text))
        per_field[label] = {"edit": e, "weight": w}
        num += w * e
        den += w
    for nl, orig in sub_norm.items():
        in_ref = any(norm_label(k) == nl for k in ref_f)
        if not in_ref and nl != "_UNLABELLED_":
            e = weighted_edit([], normalize_text(sub_f[orig]))
            per_field[f"EXTRA:{orig}"] = {"edit": e, "weight": 3.0}
            num += 3.0 * e
            den += 3.0
    f = num / den if den else 0.0
    imp = weighted_edit(normalize_text(ref_i), normalize_text(sub_i))
    return {"F": f, "I": imp, "RES": 0.65 * f + 0.35 * imp,
            "fields": per_field}


def mean_res(cases: list[dict]) -> float:
    """Average RES over a list of res_case dicts, return float."""
    if not cases:
        return 0.0
    return sum(c["RES"] for c in cases) / len(cases)
