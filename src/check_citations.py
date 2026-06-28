"""Citation-drift guard: verify every codefile `path ::symbol` still resolves.

Each lesson's `.codefile` cites a real SGLang source location as
`<span class="path">repo/relative/file.py ::Symbol</span>`. As SGLang evolves,
files move and symbols get renamed/removed (this guide pins a checkout). This
script walks every cited `path ::symbol`, resolves the file under the SGLang
source tree, and confirms the symbol still exists -- so drift is caught
mechanically instead of by eye.

Usage:
    python check_citations.py [SGLANG_ROOT]
    # default SGLANG_ROOT = ~/course/sglang  (or $SGLANG_ROOT)

Symbol forms understood:
    Foo                -> `class Foo` / `def Foo` / `Foo = ...` (module level)
    Foo.bar            -> `class Foo`, then `def bar(` inside Foo's body
    Foo._private       -> same
A citation with no ` ::` (e.g. a `.yaml`/`.md` file) only checks file existence.

Exits non-zero if any citation does not resolve. This is a maintenance/CI
helper; it needs the SGLang source present and is NOT part of the site build.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from registry import CONTENT  # noqa: E402

DEFAULT_ROOT = os.environ.get(
    "SGLANG_ROOT", os.path.expanduser("~/course/sglang")
)

PATH_RE = re.compile(r'<span class="path">([^<]+?)</span>')


def collect_citations():
    """Return {(path, symbol): [lessons...]} across all lessons (zh side; en
    mirrors). De-duplicated so each location is checked once."""
    cites = {}
    for fname, c in CONTENT.items():
        text = c.get("zh", "") + c.get("en", "")
        for raw in PATH_RE.findall(text):
            raw = raw.strip()
            if " ::" in raw:
                path, symbol = raw.split(" ::", 1)
                path, symbol = path.strip(), symbol.strip()
            else:
                path, symbol = raw.strip(), None
            cites.setdefault((path, symbol), set()).add(fname)
    return cites


def _class_body(src, cls):
    """Return the source slice of top-level `class cls`'s body, or None.
    Handles multi-line base-class signatures (Scheduler(...) spans lines)."""
    m = re.search(rf"^class {re.escape(cls)}\b", src, re.M)
    if not m:
        return None
    # Walk to the ':' that closes the (possibly multi-line) signature.
    depth, j, n = 0, m.end(), len(src)
    while j < n:
        ch = src[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            break
        j += 1
    nl = src.find("\n", j)
    if nl < 0:
        return None
    body_start = nl + 1
    # Body runs until the next top-level (column-0, non-space) construct.
    nxt = re.search(r"^\S", src[body_start:], re.M)
    body_end = body_start + nxt.start() if nxt else n
    return src[body_start:body_end]


def symbol_exists(src, symbol):
    if "." in symbol:
        cls, attr = symbol.split(".", 1)
        body = _class_body(src, cls)
        if body is None:
            return False
        # nested attr (Foo.Bar.baz) -> just check the last component is def'd
        leaf = attr.split(".")[-1]
        return bool(
            re.search(rf"^\s*(?:async\s+)?def {re.escape(leaf)}\b", body, re.M)
            or re.search(rf"^\s*{re.escape(leaf)}\s*[:=]", body, re.M)
        )
    # module-level class / function / assignment (enum, dataclass instance, ...)
    return bool(
        re.search(rf"^class {re.escape(symbol)}\b", src, re.M)
        or re.search(rf"^\s*(?:async\s+)?def {re.escape(symbol)}\b", src, re.M)
        or re.search(rf"^{re.escape(symbol)}\s*[:=]", src, re.M)
    )


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    if not os.path.isdir(root):
        print(f"SGLANG_ROOT not found: {root}\n"
              f"Pass the SGLang source path: python check_citations.py /path/to/sglang")
        return 2

    cites = collect_citations()
    bad_path, bad_symbol, ok = [], [], 0
    for (path, symbol), lessons in sorted(cites.items()):
        where = ", ".join(sorted(lessons))
        full = os.path.join(root, path)
        if not os.path.isfile(full):
            bad_path.append((path, symbol, where))
            continue
        if symbol is None:
            ok += 1
            continue
        try:
            src = open(full, encoding="utf-8", errors="replace").read()
        except OSError as e:
            bad_path.append((path, f"{symbol} (read error: {e})", where))
            continue
        if symbol_exists(src, symbol):
            ok += 1
        else:
            bad_symbol.append((path, symbol, where))

    for path, symbol, where in bad_path:
        sym = f" ::{symbol}" if symbol else ""
        print(f"  [PATH ] {path}{sym}  (cited in {where})")
    for path, symbol, where in bad_symbol:
        print(f"  [SYMB ] {path} ::{symbol}  -> symbol not found  (cited in {where})")

    total = len(cites)
    drift = len(bad_path) + len(bad_symbol)
    print(f"\nChecked {total} distinct citations against {root}: "
          f"{ok} ok, {len(bad_path)} missing file(s), {len(bad_symbol)} missing symbol(s).")
    if drift:
        print("citation drift DETECTED")
        return 1
    print("all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
