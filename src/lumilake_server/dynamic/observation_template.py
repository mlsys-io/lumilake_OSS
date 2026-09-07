"""Source template for the observation LambdaOp.

This file is the literal source of the observation function that runs inside
the server's restricted Lambda sandbox. It is loaded as text and interpolated
(the ``{width}`` placeholder becomes the preview character budget) before being
embedded in the round graph's ``LambdaOp._code``.

The sandbox injects ``json`` as a global (never imported) and whitelists only a
few builtins. ``json.JSONDecodeError`` is attribute access on the whitelisted
``json`` module, so it is catchable inside the sandbox; malformed or plain-text
input is caught and treated as text rather than raising.
"""


def observe(args):
    # ``args`` is one entry per leaf. Each entry may be a JSON string, a list of
    # JSON strings (archived leaf outputs), a list of records, or plain text.
    # Normalize every entry into a flat list of records so the numeric stats
    # below see real values.
    def unwrap(value):
        # Retrieval leaves arrive as {"df": "<json string>"} whose value is
        # itself a column-oriented table; unwrap the envelope so the transpose
        # below sees {column: {row_index: value}}.
        if (
            isinstance(value, dict)
            and list(value) == ["df"]
            and isinstance(value["df"], str)
        ):
            try:
                return json.loads(value["df"])  # fmt: skip  # noqa: E501, F821  # type: ignore[name-defined]
            except json.JSONDecodeError:  # noqa: F821  # type: ignore[name-defined]
                return value
        return value

    def transpose(value):
        # Column-oriented table: {column: {row_index: value}}. Transpose to
        # row-oriented records so each record comes from one table.
        records = {}
        for col, idxvals in value.items():
            if not isinstance(idxvals, dict):
                continue
            for idx, val in idxvals.items():
                rec = records.get(idx)
                if rec is None:
                    rec = {}
                    records[idx] = rec
                rec[col] = val
        return list(records.values())

    def records(value):
        # Normalize one item into a flat list of row-oriented records. Each
        # table is transposed on its own so records never mix columns across
        # leaves.
        value = unwrap(value)
        vals = list(value.values()) if isinstance(value, dict) else []
        if vals and isinstance(vals[0], dict):
            return transpose(value)
        return [value]

    rows = []
    for entry in args:
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)  # noqa: F821  # type: ignore[name-defined]
            except json.JSONDecodeError:  # noqa: F821  # type: ignore[name-defined]
                rows.append(entry)
                continue
        if isinstance(entry, list):
            for item in entry:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)  # fmt: skip  # noqa: E501, F821  # type: ignore[name-defined]
                    except json.JSONDecodeError:  # fmt: skip  # noqa: E501, F821  # type: ignore[name-defined]
                        pass
                rows.extend(records(item))
        else:
            rows.extend(records(entry))
    n = len(rows)
    cols = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        for k in r:
            if isinstance(r[k], (int, float)) and k not in cols:
                cols.append(k)
    sums = {}
    mins = {}
    maxs = {}
    for k in cols:
        vals = [
            float(r[k])
            for r in rows
            if isinstance(r, dict) and k in r and isinstance(r[k], (int, float))
        ]
        if vals:
            sums[k] = sum(vals)
            mins[k] = min(vals)
            maxs[k] = max(vals)
    lines = [f"rows={n}"]
    for k in cols:
        if k in sums:
            cnt = len(
                [
                    1
                    for r in rows
                    if isinstance(r, dict) and k in r and isinstance(r[k], (int, float))
                ]
            )
            lines.append(
                f"{k}: n={cnt} sum={sums[k]:.4g} min={mins[k]:.4g} "
                f"max={maxs[k]:.4g} avg={sums[k] / max(1, cnt):.4g}"
            )
    preview = json.dumps(rows[:3], ensure_ascii=False)  # fmt: skip  # noqa: E501, F821  # type: ignore[name-defined]
    if len(preview) > {width}:  # noqa: F821  # type: ignore[name-defined]
        preview = preview[:{width}] + "..."  # noqa: F821  # type: ignore[name-defined]
    lines.append("preview=" + preview)
    return "\n".join(lines)
