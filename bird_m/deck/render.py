"""BIRD-Deck -> standalone HTML (16:9, 1280x720 per slide), charts as inline SVG.

The model only chooses content and layouts; this renderer owns the design
(typography, grid, colour, data-viz), so every valid deck looks professional.
The same HTML is the base of the web editor and of the visual reward loop
(scripts/deck_check.mjs screenshots each slide and measures overflow).
"""

from __future__ import annotations

import math
import re
from html import escape

W, H = 1280, 720
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

PRESETS = {
    "bird": {"primary": "#1E2A78", "accent": "#FF6B4A"},
    "ardoise": {"primary": "#2F3E46", "accent": "#52B788"},
    "foret": {"primary": "#1B4332", "accent": "#E9A23B"},
    "corail": {"primary": "#7A1F3D", "accent": "#F28C28"},
}

I18N = {
    "fr": {"recommended": "Recommandé", "source": "Source"},
    "en": {"recommended": "Recommended", "source": "Source"},
}


# ----------------------------------------------------------------- helpers
def e(s) -> str:
    return escape(str(s), quote=True)


def fmt_num(v: float, lang: str, unit: str = "") -> str:
    if float(v).is_integer():
        s = f"{int(v):,}"
    else:
        s = f"{v:,.1f}"
    if lang == "fr":
        s = s.replace(",", " ").replace(".", ",")
    if unit:
        s += (" " if lang == "fr" and unit[0] in "%€$kKM" else "") + unit
    return s


def _color(value: str | None, fallback: str) -> str:
    return value if isinstance(value, str) and _HEX.match(value) else fallback


def _nb(v: str) -> str:
    """Keep '18,4 M€' on one line."""
    return v.replace(" ", "\u202f")


def _trend(t: str | None) -> str:
    return {"up": "▲", "down": "▼", "flat": "▶"}.get(t or "", "")


# ----------------------------------------------------------------- charts (SVG)
def _palette(n: int) -> list[str]:
    base = ["var(--primary)", "var(--accent)", "var(--primary-50)", "var(--muted-2)"]
    return base[:n]


def _scale(lo: float, hi: float, top: float, bottom: float):
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi == lo:
        hi = lo + 1
    return lambda v: bottom - (v - lo) / (hi - lo) * (bottom - top), bottom - (
        0 - lo
    ) / (hi - lo) * (bottom - top)


def _legend(series: list[dict], colors: list[str], x: float, y: float) -> str:
    out, cx = [], x
    for s, c in zip(series, colors):
        out.append(
            f'<rect x="{cx}" y="{y - 10}" width="12" height="12" rx="3" fill="{c}"/>'
        )
        out.append(f'<text x="{cx + 18}" y="{y}" class="legend">{e(s["name"])}</text>')
        cx += 30 + 9 * len(s["name"])
    return "".join(out)


def chart_svg(ch: dict, lang: str, w: int = 720, h: int = 430) -> str:
    kind, cats, series = ch["type"], ch["categories"], ch["series"]
    unit, hl = ch.get("unit", ""), ch.get("highlight")
    parts: list[str] = []
    if kind == "donut":
        vals = series[0]["values"]
        total = sum(vals) or 1
        cx, cy, r, sw = 200, h / 2, 150, 54
        a0 = -math.pi / 2
        colors = [
            "var(--primary)",
            "var(--accent)",
            "var(--primary-50)",
            "var(--muted-2)",
            "var(--accent-50)",
            "var(--primary-25)",
        ]
        for i, v in enumerate(vals):
            frac = v / total
            a1 = a0 + frac * 2 * math.pi
            if frac >= 0.9999:
                parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{colors[i]}" stroke-width="{sw}"/>'
                )
            elif frac > 0:
                large = 1 if frac > 0.5 else 0
                x0, y0 = cx + r * math.cos(a0), cy + r * math.sin(a0)
                x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
                parts.append(
                    f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f}" fill="none" '
                    f'stroke="{colors[i]}" stroke-width="{sw}"/>'
                )
            a0 = a1
        focus = hl if hl is not None else max(range(len(vals)), key=lambda i: vals[i])
        pct = fmt_num(round(vals[focus] / total * 100), lang, "%")
        parts.append(
            f'<text x="{cx}" y="{cy + 6}" class="donut-value" text-anchor="middle">{pct}</text>'
        )
        parts.append(
            f'<text x="{cx}" y="{cy + 40}" class="donut-label" text-anchor="middle">{e(cats[focus])}</text>'
        )
        ly = h / 2 - len(vals) * 22
        for i, (c, v) in enumerate(zip(cats, vals)):
            y = ly + i * 44
            parts.append(
                f'<rect x="420" y="{y - 14}" width="16" height="16" rx="4" fill="{colors[i]}"/>'
            )
            parts.append(f'<text x="446" y="{y}" class="legend-strong">{e(c)}</text>')
            parts.append(
                f'<text x="446" y="{y + 20}" class="legend">{fmt_num(v, lang, unit)} · {round(v / total * 100)} %</text>'
            )
        return f'<svg viewBox="0 0 {w} {h}" class="chart">{"".join(parts)}</svg>'

    left, right, top, bottom = 16, w - 16, 40 if len(series) > 1 else 24, h - 44
    n = len(cats)
    if kind == "stacked_bar":
        sums_pos = [sum(max(s["values"][i], 0) for s in series) for i in range(n)]
        sums_neg = [sum(min(s["values"][i], 0) for s in series) for i in range(n)]
        lo, hi = min(sums_neg), max(sums_pos)
    else:
        allv = [v for s in series for v in s["values"]]
        lo, hi = min(allv), max(allv)
    if kind == "line":  # lines show variation: zoom on the data range, baseline at the bottom
        pad = (hi - lo) * 0.25 or abs(hi) * 0.1 or 1.0
        lo_d, hi_d = lo - pad, hi + pad
        top_px, y0 = top + 18, bottom

        def y(v: float) -> float:
            return y0 - (v - lo_d) / (hi_d - lo_d) * (y0 - top_px)
    else:
        y, y0 = _scale(lo, hi * 1.12 if hi > 0 else hi, top + 18, bottom)
    colors = _palette(len(series))
    slot = (right - left) / n
    parts.append(
        f'<line x1="{left}" y1="{y0:.1f}" x2="{right}" y2="{y0:.1f}" class="baseline"/>'
    )
    for i, c in enumerate(cats):
        cls = "cat strong" if hl == i else "cat"
        parts.append(
            f'<text x="{left + slot * (i + 0.5):.1f}" y="{h - 14}" class="{cls}" text-anchor="middle">{e(c)}</text>'
        )

    if kind == "bar":
        k = len(series)
        bw = min(64, slot * 0.62 / k)
        for si, s in enumerate(series):
            for i, v in enumerate(s["values"]):
                x = left + slot * (i + 0.5) - bw * k / 2 + si * bw
                yt, yb = sorted((y(v), y0))
                fill = colors[si]
                if k == 1 and hl is not None:
                    fill = "var(--primary)" if i == hl else "var(--primary-25)"
                parts.append(
                    f'<rect x="{x + 2:.1f}" y="{yt:.1f}" width="{bw - 4:.1f}" height="{max(yb - yt, 1):.1f}" rx="5" fill="{fill}"/>'
                )
                ty = yt - 10 if v >= 0 else yb + 22
                cls = "val strong" if (hl == i or hl is None) else "val"
                parts.append(
                    f'<text x="{x + bw / 2:.1f}" y="{ty:.1f}" class="{cls}" text-anchor="middle">{fmt_num(v, lang, unit)}</text>'
                )
    elif kind == "stacked_bar":
        bw = min(88, slot * 0.6)
        for i in range(n):
            pos, neg = 0.0, 0.0
            x = left + slot * (i + 0.5) - bw / 2
            for si, s in enumerate(series):
                v = s["values"][i]
                base = pos if v >= 0 else neg
                yt, yb = sorted((y(base + v), y(base)))
                parts.append(
                    f'<rect x="{x:.1f}" y="{yt:.1f}" width="{bw:.1f}" height="{max(yb - yt, 0.5):.1f}" fill="{colors[si]}"/>'
                )
                if v >= 0:
                    pos += v
                else:
                    neg += v
            parts.append(
                f'<text x="{x + bw / 2:.1f}" y="{y(pos) - 10:.1f}" class="val strong" text-anchor="middle">{fmt_num(pos + neg, lang, unit)}</text>'
            )
    else:  # line
        all_pts = [[(left + slot * (i + 0.5), y(v)) for i, v in enumerate(sr["values"])] for sr in series]
        for si, pts in enumerate(all_pts):
            d = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py) in enumerate(pts))
            sw = 4 if si == 0 else 3
            parts.append(
                f'<path d="{d}" fill="none" stroke="{colors[si]}" stroke-width="{sw}" '
                'stroke-linejoin="round" stroke-linecap="round"/>'
            )
        marks = sorted({n - 1} | ({hl} if hl is not None else set()))
        for i in marks:
            # highest point gets its label above, the next ones below if they would collide
            order = sorted(range(len(series)), key=lambda si: all_pts[si][i][1])
            placed: list[float] = []
            for rank, si in enumerate(order):
                px, py = all_pts[si][i]
                ly = py - 18
                if rank > 0 and any(abs(ly - q) < 56 for q in placed):
                    ly = py + 32
                placed.append(ly)
                parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" fill="var(--bg)" stroke="{colors[si]}" stroke-width="4"/>')
                parts.append(
                    f'<text x="{px:.1f}" y="{ly:.1f}" class="val strong" text-anchor="middle" '
                    f'fill="{colors[si]}">{fmt_num(series[si]["values"][i], lang, unit)}</text>'
                )
    if len(series) > 1:
        parts.append(_legend(series, colors, left, 18))
    return f'<svg viewBox="0 0 {w} {h}" class="chart">{"".join(parts)}</svg>'


# ----------------------------------------------------------------- layouts
def _head(s: dict) -> str:
    k = f'<div class="kicker">{e(s["kicker"])}</div>' if s.get("kicker") else ""
    return f'<header>{k}<h2 class="fit">{e(s["title"])}</h2></header>'


def _bullets(items: list[str], cls: str = "bullets") -> str:
    dense = (
        " dense" if len(items) > 4 or sum(len(b.split()) for b in items) > 45 else ""
    )
    return (
        f'<ul class="{cls}{dense}">'
        + "".join(f"<li>{e(b)}</li>" for b in items)
        + "</ul>"
    )


def render_slide(s: dict, lang: str) -> str:
    t = I18N[lang]
    L = s["layout"]
    if L == "title":
        k = f'<div class="kicker">{e(s["kicker"])}</div>' if s.get("kicker") else ""
        sub = f'<p class="subtitle">{e(s["subtitle"])}</p>' if s.get("subtitle") else ""
        return f'<div class="hero fit">{k}<h1>{e(s["title"])}</h1>{sub}</div>'
    if L == "section":
        num = (
            f'<div class="section-num">{s["number"]:02d}</div>'
            if s.get("number")
            else ""
        )
        sub = f'<p class="subtitle">{e(s["subtitle"])}</p>' if s.get("subtitle") else ""
        return f'<div class="hero fit">{num}<h1>{e(s["title"])}</h1>{sub}</div>'
    if L == "agenda":
        cols = " two" if len(s["items"]) > 5 else ""
        rows = math.ceil(len(s["items"]) / 2)
        items = "".join(
            f'<li><span class="n">{i + 1:02d}</span><span>{e(x)}</span></li>'
            for i, x in enumerate(s["items"])
        )
        style = f' style="grid-template-rows:repeat({rows},auto)"' if cols else ""
        return _head(s) + f'<ol class="agenda fit{cols}"{style}>{items}</ol>'
    if L == "bullets":
        return _head(s) + f'<div class="body fit">{_bullets(s["bullets"])}</div>'
    if L == "two_columns":
        col = lambda c: (
            f'<div class="card"><h3>{e(c["heading"])}</h3>{_bullets(c["bullets"], "bullets small")}</div>'
        )
        return (
            _head(s) + f'<div class="grid2 fit">{col(s["left"])}{col(s["right"])}</div>'
        )
    if L == "big_number":
        tr = s.get("trend")
        arrow = f'<span class="trend {tr}">{_trend(tr)}</span>' if tr else ""
        ctx = f'<p class="context">{e(s["context"])}</p>' if s.get("context") else ""
        return _head(s) + (
            f'<div class="bignum fit"><div class="value">{e(_nb(s["value"]))}{arrow}</div>'
            f'<div class="label">{e(s["label"])}</div>{ctx}</div>'
        )
    if L == "kpis":
        cards = ""
        for k in s["kpis"]:
            tr = k.get("trend")
            delta = (
                f'<div class="delta {tr or ""}">{_trend(tr)} {e(k["delta"])}</div>'
                if k.get("delta")
                else ""
            )
            cards += f'<div class="kpi"><div class="kv">{e(_nb(k["value"]))}</div><div class="kl">{e(k["label"])}</div>{delta}</div>'
        return _head(s) + f'<div class="kpis n{len(s["kpis"])} fit">{cards}</div>'
    if L == "chart":
        src = (
            f'<div class="source">{t["source"]} : {e(s["source"])}</div>'
            if s.get("source")
            else ""
        )
        return _head(s) + (
            f'<div class="chartwrap fit"><div class="plot">{chart_svg(s["chart"], lang)}</div>'
            f'<aside class="takeaway"><p>{e(s["takeaway"])}</p>{src}</aside></div>'
        )
    if L == "comparison":
        cards = ""
        for o in s["options"]:
            rec = (
                f'<div class="badge">{t["recommended"]}</div>'
                if o.get("recommended")
                else ""
            )
            cls = "card rec" if o.get("recommended") else "card"
            cards += f'<div class="{cls}">{rec}<h3>{e(o["heading"])}</h3>{_bullets(o["points"], "bullets small")}</div>'
        verdict = (
            f'<div class="verdict">{e(s["verdict"])}</div>' if s.get("verdict") else ""
        )
        return (
            _head(s)
            + f'<div class="cmp n{len(s["options"])} fit">{cards}</div>{verdict}'
        )
    if L == "timeline":
        steps = ""
        for st in s["steps"]:
            cls = "step done" if st.get("done") else "step"
            det = (
                f'<div class="detail">{e(st["detail"])}</div>'
                if st.get("detail")
                else ""
            )
            steps += f'<div class="{cls}"><div class="date">{e(st["date"])}</div><div class="dot"></div><div class="slabel">{e(st["label"])}</div>{det}</div>'
        return (
            _head(s)
            + f'<div class="timeline n{len(s["steps"])} fit">{steps}</div>'
        )
    if L == "quote":
        role = f'<span class="role"> · {e(s["role"])}</span>' if s.get("role") else ""
        return f'<figure class="quote fit"><div class="qmark">“</div><blockquote>{e(s["quote"])}</blockquote><figcaption><strong>{e(s["author"])}</strong>{role}</figcaption></figure>'
    if L == "table":
        hl = s.get("highlight_row")
        head = "".join(f"<th>{e(h)}</th>" for h in s["header"])
        rows = "".join(
            f"<tr{' class=hl' if i == hl else ''}>"
            + "".join(f"<td>{e(c)}</td>" for c in r)
            + "</tr>"
            for i, r in enumerate(s["rows"])
        )
        return (
            _head(s)
            + f'<div class="body fit"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
        )
    if L == "closing":
        bl = _bullets(s["bullets"], "bullets light") if s.get("bullets") else ""
        cta = f'<div class="cta">{e(s["cta"])}</div>' if s.get("cta") else ""
        contact = (
            f'<div class="contact">{e(s["contact"])}</div>' if s.get("contact") else ""
        )
        return f'<div class="hero fit"><h1>{e(s["title"])}</h1>{bl}{cta}{contact}</div>'
    raise ValueError(f"unknown layout {L!r}")


DARK = {"title", "section", "closing"}

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{background:#d9dbe1;font-family:Inter,'Helvetica Neue',Arial,sans-serif;color:var(--text);-webkit-font-smoothing:antialiased}
.slide{position:relative;width:1280px;height:720px;overflow:hidden;background:var(--bg);margin:24px auto;padding:64px 80px 72px;display:flex;flex-direction:column;gap:36px}
.slide.dark{background:var(--primary);color:#fff;justify-content:center}
.slide.dark::after{content:"";position:absolute;right:-180px;top:-180px;width:560px;height:560px;border-radius:50%;background:radial-gradient(circle,var(--accent-40),transparent 70%)}
.slide:not(.dark)::before{content:"";position:absolute;left:0;top:0;width:100%;height:6px;background:linear-gradient(90deg,var(--primary) 0 70%,var(--accent) 70%)}
header{display:flex;flex-direction:column;gap:12px}
.kicker{font-size:15px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
h1{font-family:'Inter Display',Inter,sans-serif;font-size:66px;line-height:1.05;font-weight:700;letter-spacing:-.03em;max-width:980px}
h2{font-family:'Inter Display',Inter,sans-serif;font-size:38px;line-height:1.15;font-weight:700;letter-spacing:-.02em;max-height:88px;overflow:hidden;color:var(--primary-ink)}
h3{font-size:22px;font-weight:700;color:var(--primary);letter-spacing:-.01em}
.hero{position:relative;z-index:1;display:flex;flex-direction:column;gap:22px}
.subtitle{font-size:26px;line-height:1.4;opacity:.82;max-width:900px;font-weight:400}
.section-num{font-family:'Inter Display',Inter;font-size:120px;font-weight:800;line-height:1;color:transparent;-webkit-text-stroke:2px var(--accent)}
.body{flex:1;min-height:0}
.bullets{list-style:none;display:flex;flex-direction:column;gap:22px;font-size:27px;line-height:1.38}
.bullets.dense{gap:16px;font-size:23px}
.bullets li{position:relative;padding-left:34px}
.bullets li::before{content:"";position:absolute;left:0;top:.55em;width:14px;height:4px;border-radius:2px;background:var(--accent)}
.bullets.small{font-size:20px;gap:12px}.bullets.small.dense{font-size:19px;gap:10px}
.bullets.light{font-size:24px;opacity:.9;margin-top:4px}
.grid2{flex:1;min-height:0;display:grid;grid-template-columns:1fr 1fr;gap:32px}
.card{position:relative;background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:30px 32px;display:flex;flex-direction:column;gap:18px;min-height:0}
.agenda{list-style:none;flex:1;min-height:0;display:grid;grid-template-columns:1fr;align-content:start;gap:16px 56px;font-size:28px;font-weight:500}
.agenda.two{grid-template-columns:1fr 1fr;grid-auto-flow:column}
.L-agenda .agenda{align-content:center;font-size:30px;gap:20px 56px}
.agenda li{display:flex;gap:22px;align-items:baseline;border-bottom:1px solid var(--line);padding-bottom:16px}
.agenda .n{font-family:'Inter Display',Inter;font-weight:700;color:var(--accent);font-size:24px;min-width:40px}
.bignum{flex:0 0 auto;margin:auto 0;display:flex;flex-direction:column;justify-content:center;gap:14px;border-left:6px solid var(--accent);padding-left:44px}
.bignum .value{white-space:nowrap;font-family:'Inter Display',Inter;font-size:168px;line-height:.95;font-weight:800;letter-spacing:-.04em;color:var(--primary);display:flex;align-items:flex-start;gap:20px}
.bignum .label{font-size:32px;font-weight:600;max-width:900px}
.bignum .context{font-size:22px;line-height:1.45;color:var(--muted);max-width:880px}
.trend{font-size:44px;margin-top:22px}.trend.up{color:var(--good)}.trend.down{color:var(--bad)}.trend.flat{color:var(--muted)}
.kpis{flex:0 0 auto;margin:auto 0;display:grid;gap:24px;align-items:stretch}.kpis.n2{grid-template-columns:repeat(2,1fr)}.kpis.n3{grid-template-columns:repeat(3,1fr)}.kpis.n4{grid-template-columns:repeat(4,1fr)}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:40px 30px;min-height:250px;display:flex;flex-direction:column;justify-content:center;gap:12px}
.kv{font-family:'Inter Display',Inter;white-space:nowrap;font-variant-numeric:tabular-nums;font-size:64px;font-weight:800;letter-spacing:-.03em;color:var(--primary);line-height:1}
.n3 .kv{font-size:58px}.n4 .kv{font-size:46px}
.kl{font-size:21px;color:var(--muted);font-weight:500;line-height:1.3}
.delta{align-self:flex-start;font-size:17px;font-weight:700;padding:6px 12px;border-radius:999px;background:var(--line);color:var(--muted)}
.delta.up{background:var(--good-bg);color:var(--good)}.delta.down{background:var(--bad-bg);color:var(--bad)}
.chartwrap{flex:1;min-height:0;display:grid;grid-template-columns:1fr 330px;gap:40px;align-items:center}
.plot{height:100%;min-height:0;display:flex;align-items:center}
.chart{width:100%;height:100%;max-height:440px}
.chart text{font-family:Inter,sans-serif}
.chart .cat{font-size:17px;fill:var(--muted)}.chart .cat.strong{fill:var(--text);font-weight:700}
.chart .val{font-size:17px;fill:var(--muted);font-weight:600}.chart .val.strong{fill:var(--text);font-weight:700}
.chart .baseline{stroke:var(--line-strong);stroke-width:2}
.chart .legend{font-size:15px;fill:var(--muted)}.chart .legend-strong{font-size:18px;fill:var(--text);font-weight:600}
.chart .donut-value{font-family:'Inter Display',Inter;font-size:52px;font-weight:800;fill:var(--primary)}.chart .donut-label{font-size:17px;fill:var(--muted)}
.takeaway{border-left:5px solid var(--accent);padding:6px 0 6px 28px;display:flex;flex-direction:column;gap:18px}
.takeaway p{font-size:27px;line-height:1.35;font-weight:600;color:var(--primary-ink)}
.source{font-size:14px;color:var(--muted)}
.cmp{flex:0 1 auto;margin:auto 0;min-height:0;display:grid;gap:28px}.cmp .card{min-height:250px}.cmp.n2{grid-template-columns:1fr 1fr}.cmp.n3{grid-template-columns:repeat(3,1fr)}
.card.rec{border:2px solid var(--primary);background:var(--primary-5);box-shadow:0 12px 32px -18px var(--primary)}
.badge{position:absolute;top:-14px;left:28px;background:var(--primary);color:#fff;font-size:13px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:6px 12px;border-radius:999px}
.verdict{font-size:22px;font-weight:600;color:var(--primary-ink);background:var(--accent-15);border-radius:12px;padding:16px 24px}
.timeline{position:relative;flex:1;display:grid;gap:24px;align-content:center}
.timeline.n3{grid-template-columns:repeat(3,1fr)}.timeline.n4{grid-template-columns:repeat(4,1fr)}.timeline.n5{grid-template-columns:repeat(5,1fr)}.timeline.n6{grid-template-columns:repeat(6,1fr)}
.step{position:relative;display:flex;flex-direction:column;gap:0}
.step::before{content:"";position:absolute;left:0;right:-24px;top:61px;height:3px;background:var(--line-strong)}
.step:last-of-type::before{right:0}
.step.done::before{background:var(--primary)}
.step .date{height:36px;display:flex;align-items:flex-end;font-size:18px;font-weight:700;color:var(--accent);letter-spacing:.04em}
.step .dot{margin-top:12px;width:28px;height:28px;border-radius:50%;background:var(--bg);border:4px solid var(--primary);z-index:1;position:relative}
.step.done .dot{background:var(--primary)}
.step .slabel{margin-top:22px;font-size:23px;font-weight:700;line-height:1.25;color:var(--primary-ink);padding-right:16px}
.step .detail{margin-top:10px;font-size:17px;line-height:1.45;color:var(--muted);padding-right:16px}
.quote{flex:1;display:flex;flex-direction:column;justify-content:center;gap:26px;padding:0 40px}
.qmark{font-family:Georgia,serif;font-size:160px;line-height:.6;color:var(--accent);height:70px}
blockquote{font-family:'Inter Display',Inter;font-size:40px;line-height:1.3;font-weight:500;letter-spacing:-.015em;color:var(--primary-ink)}
figcaption{font-size:21px;color:var(--muted)}figcaption strong{color:var(--text)}
.L-table .body{display:flex;align-items:center}
table{width:100%;font-variant-numeric:tabular-nums;border-collapse:collapse;font-size:21px}
th{text-align:left;font-size:14px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);padding:0 18px 14px;border-bottom:2px solid var(--line-strong)}
td{padding:16px 18px;border-bottom:1px solid var(--line)}
td:not(:first-child),th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}
tr.hl td{background:var(--primary-5);font-weight:700;color:var(--primary-ink)}
.cta{align-self:flex-start;background:var(--accent);color:#fff;font-size:24px;font-weight:700;padding:16px 28px;border-radius:999px;margin-top:8px}
.contact{font-size:20px;opacity:.75}
footer{position:absolute;left:80px;right:80px;bottom:28px;display:flex;justify-content:space-between;font-size:14px;color:var(--muted);letter-spacing:.02em}
footer .brand{font-weight:700;color:var(--primary)}
"""


def _theme_vars(theme: dict) -> str:
    preset = PRESETS.get(theme.get("preset", "bird"), PRESETS["bird"])
    p = _color(theme.get("primary"), preset["primary"])
    a = _color(theme.get("accent"), preset["accent"])
    mix = lambda c, pct, base="white": f"color-mix(in srgb, {c} {pct}%, {base})"
    return (
        f":root{{--primary:{p};--accent:{a};--bg:#FBFAF7;--surface:#FFFFFF;--text:#16181D;--muted:#5F6673;--muted-2:#A3A9B5;"
        f"--line:#E7E5DF;--line-strong:#CFCCC4;--good:#1A7F4B;--good-bg:#E3F4EA;--bad:#B42318;--bad-bg:#FDE8E6;"
        f"--primary-ink:{mix(p, 85, '#111')};--primary-50:{mix(p, 50)};--primary-25:{mix(p, 25)};--primary-5:{mix(p, 6)};"
        f"--accent-50:{mix(a, 50)};--accent-40:{mix(a, 45, 'transparent')};--accent-15:{mix(a, 15)}}}"
    )


def render_deck(deck: dict) -> str:
    lang = deck["meta"]["lang"]
    theme = deck.get("theme", {})
    brand = theme.get("brand", "")
    n = len(deck["slides"])
    sections = []
    for i, s in enumerate(deck["slides"]):
        dark = s["layout"] in DARK
        footer = ""
        if not dark:
            left = f'<span class="brand">{e(brand)}</span> · ' if brand else ""
            footer = f"<footer><span>{left}{e(deck['meta']['title'])}</span><span>{i + 1} / {n}</span></footer>"
        cls = f"slide L-{s['layout']}{' dark' if dark else ''}"
        sections.append(
            f'<section class="{cls}" data-index="{i}">{render_slide(s, lang)}{footer}</section>'
        )
    return (
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
        f"<title>{e(deck['meta']['title'])}</title><style>{_theme_vars(theme)}{CSS}</style></head>"
        f"<body>{''.join(sections)}</body></html>"
    )
