"""Build ``outputs/tag_weather.html``: a thumbnail grid for manually tagging MWPD weather/lighting.

Open the HTML in a browser, tag every image, click "Export CSV" and save the file as
``data/processed/mwpd/weather_tags.csv`` (columns: ``split,file,weather,lighting``).
Progress is autosaved in the browser's localStorage.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from src.data.inspect import LIGHTING_PATTERNS, inspect_mwpd_conditions, match_condition
from src.utils.common import base_parser, imread, load_config, project_path, setup_logging

log = logging.getLogger("tag_weather")

HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>MWPD weather tagging</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f4}
header{position:sticky;top:0;background:#222;color:#fff;padding:8px 12px;z-index:9;
display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header button,header select{font-size:14px}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;padding:8px}
.card{background:#fff;border:4px solid #bbb;border-radius:6px;padding:4px;font-size:12px}
.card.normal{border-color:#0072B2}.card.rainy{border-color:#D55E00}
.card img{width:100%;aspect-ratio:1;object-fit:cover;cursor:pointer;display:block}
.row{display:flex;gap:6px;margin-top:3px;flex-wrap:wrap}
.name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#555}
.hidden{display:none}
</style></head><body>
<header>
<b>MWPD weather tagging</b>
<span id="progress"></span>
<label>Show <select id="filter"><option value="all">all</option>
<option value="untagged">untagged</option><option value="normal">normal</option>
<option value="rainy">rainy</option></select></label>
<button id="bulk">Set visible untagged &rarr; normal</button>
<button id="export">Export CSV</button>
<span>Click image = toggle normal/rainy. Blue = normal, orange = rainy.</span>
</header>
<div id="grid"></div>
<script>
const ITEMS = __ITEMS__;
const KEY = "mwpd_weather_tags___STAMP__";
let tags = {};
try { tags = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { tags = {}; }
const id = it => it.split + "/" + it.file;
ITEMS.forEach(it => {
  tags[id(it)] = tags[id(it)] || {weather: it.weather, lighting: it.lighting};
});
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(tags)); } catch (e) {} };
const grid = document.getElementById("grid");
function radios(key, field, opts) {
  return opts.map(o => `<label><input type="radio" name="${field}:${key}" value="${o}"
    ${tags[key][field] === o ? "checked" : ""}>${o}</label>`).join("");
}
ITEMS.forEach(it => {
  const key = id(it), c = document.createElement("div");
  c.className = "card"; c.dataset.key = key;
  c.innerHTML = `<img loading="lazy" src="${it.src}" title="${key}">
    <div class="name">${key}</div>
    <div class="row">${radios(key, "weather", ["normal", "rainy"])}</div>
    <div class="row">${radios(key, "lighting", ["day", "twilight", "night"])}</div>`;
  grid.appendChild(c);
});
function paint(card) {
  const w = tags[card.dataset.key].weather;
  card.classList.toggle("normal", w === "normal"); card.classList.toggle("rainy", w === "rainy");
  card.querySelectorAll("input").forEach(i => {
    i.checked = tags[card.dataset.key][i.name.split(":")[0]] === i.value; });
}
function refresh() {
  const f = document.getElementById("filter").value; let n = 0;
  document.querySelectorAll(".card").forEach(c => {
    const w = tags[c.dataset.key].weather; if (w) n++; paint(c);
    c.classList.toggle("hidden", !(f === "all" || (f === "untagged" ? !w : w === f)));
  });
  document.getElementById("progress").textContent = `${n} / ${ITEMS.length} tagged`;
  save();
}
grid.addEventListener("click", e => {
  const card = e.target.closest(".card"); if (!card) return; const t = tags[card.dataset.key];
  if (e.target.tagName === "IMG") t.weather = t.weather === "normal" ? "rainy" : "normal";
  if (e.target.tagName === "INPUT") t[e.target.name.split(":")[0]] = e.target.value;
  refresh();
});
document.getElementById("filter").onchange = refresh;
document.getElementById("bulk").onclick = () => {
  document.querySelectorAll(".card:not(.hidden)").forEach(c => {
    if (!tags[c.dataset.key].weather) tags[c.dataset.key].weather = "normal"; });
  refresh();
};
document.getElementById("export").onclick = () => {
  const rows = ["split,file,weather,lighting,tagged_by"].concat(ITEMS.map(it => {
    const t = tags[id(it)];
    return [it.split, it.file, t.weather, t.lighting, "human_reviewed"].join(","); }));
  const missing = ITEMS.filter(it => !tags[id(it)].weather).length;
  if (missing && !confirm(missing + " images are untagged. Export anyway?")) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\\n") + "\\n"], {type: "text/csv"}));
  a.download = "weather_tags.csv"; a.click();
};
refresh();
</script></body></html>
"""


def thumbnail_data_uri(img: Path, size: int = 256) -> str:
    """Encode a downscaled JPEG thumbnail as a data URI so the page is self-contained.

    Args:
        img: Image file.
        size: Longest side of the thumbnail in pixels.

    Returns:
        ``data:image/jpeg;base64,...`` string.
    """
    im = imread(img)
    scale = size / max(im.shape[:2])
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_html(mwpd_root: Path, out_html: Path, prefill: Path | None = None) -> int:
    """Write the tagging page.

    Args:
        mwpd_root: Raw MWPD root.
        out_html: Destination HTML file.
        prefill: Optional existing ``weather_tags.csv`` whose tags pre-fill the page for review.

    Returns:
        Number of images listed.
    """
    cond = inspect_mwpd_conditions(mwpd_root)
    paths = {(p.parent.parent.name, p.name): p for p in mwpd_root.rglob("images/*")}
    pre: dict[tuple[str, str], dict[str, str]] = {}
    if prefill is not None and prefill.exists():
        df = pd.read_csv(prefill)
        pre = {
            (r.split, r.file): {"weather": r.weather, "lighting": r.lighting}
            for r in df.itertuples()
        }
    items = []
    for row in cond.itertuples():
        img = paths[(row.split, row.file)]
        src = thumbnail_data_uri(img)
        lighting = match_condition(row.file, LIGHTING_PATTERNS)
        items.append(
            {
                "split": row.split,
                "file": row.file,
                "src": src,
                "weather": pre.get((row.split, row.file), {}).get("weather", ""),
                "lighting": pre.get((row.split, row.file), {}).get(
                    "lighting", "day" if lighting == "unknown" else lighting
                ),
            }
        )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.replace("__ITEMS__", json.dumps(items)).replace(
        "__STAMP__", datetime.now().strftime("%Y%m%d%H%M%S")
    )
    out_html.write_text(html, encoding="utf-8")
    return len(items)


def main() -> None:
    """CLI entry point."""
    parser = base_parser(__doc__ or "")
    args = parser.parse_args()
    setup_logging("tag_weather")
    cfg = load_config("data")
    out_html = project_path(cfg["paths"]["outputs"]) / "tag_weather.html"
    if args.dry_run:
        log.info("Would write %s", out_html)
        return
    tags = project_path(cfg["paths"]["processed_mwpd"]) / "weather_tags.csv"
    n = build_html(project_path(cfg["paths"]["raw_mwpd"]), out_html, tags)
    if tags.exists():
        log.info("Pre-filled from %s - review and re-export to overwrite it", tags)
    log.info(
        "Wrote %s with %d images (lighting pre-filled: 'night' from file names, else 'day')",
        out_html,
        n,
    )
    log.info(
        "Save the exported CSV as %s",
        project_path(cfg["paths"]["processed_mwpd"]) / "weather_tags.csv",
    )


if __name__ == "__main__":
    main()
