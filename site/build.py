"""Compile the write-up into the GitHub Pages site.

    uv run --with markdown python site/build.py            # build into site/_build/
    uv run --with markdown python site/build.py --serve    # build, then serve on http://127.0.0.1:8000

Pipeline:
  docs/write-up-2.md  --markdown-->  HTML body
    · the first "# heading" becomes the page title
    · an image or table followed by a paragraph that is all italics becomes a <figure> with that caption
    · a paragraph containing only {{figure:NAME}} is replaced by the inline SVG from site/figures.py
    · links to ../data/write-up/<folder>/ point to a static Inspect viewer bundled at logs/<folder>/
  site/template.html + site/style.css wrap the body; images are copied from docs/images/.
"""
import argparse, datetime, html, re, shutil, subprocess, sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import FIGURES, static_svg

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SRC = ROOT / "docs" / "write-up-2.md"
OUT = SITE / "_build"
REPO = "https://github.com/foogunlana/emotion-concepts"
DATE = "September 2026"


def to_html(md_text: str) -> str:
    return markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "attr_list",
                                                  "toc"], extension_configs={"toc": {"permalink": False}})


def figures_and_captions(body: str) -> str:
    caption = r'\s*<p><em>((?:(?!</p>).)*?)</em>((?:(?!</p>).)*?)</p>'
    # chart placeholders (+ optional italic caption)
    def chart(m):
        svg = FIGURES[m.group(1)]()
        cap = f"<figcaption>{m.group(2)}{m.group(3) or ''}</figcaption>" if m.group(2) else ""
        return f'<figure class="chart">{svg}{cap}</figure>'
    body = re.sub(r"<p>\{\{figure:(\w+)\}\}</p>(?:" + caption + ")?", chart, body, flags=re.S)
    # images + caption
    body = re.sub(r"<p>(<img [^>]+>)</p>(?:" + caption + ")?",
                  lambda m: f"<figure>{m.group(1)}" + (f"<figcaption>{m.group(2)}{m.group(3)}</figcaption>" if m.group(2) else "")
                  + "</figure>", body, flags=re.S)
    # tables + caption
    body = re.sub(r"(<table>.*?</table>)(?:" + caption + ")?",
                  lambda m: f'<figure><div class="table-wrap">{m.group(1)}</div>'
                  + (f"<figcaption>{m.group(2)}{m.group(3)}</figcaption>" if m.group(2) else "") + "</figure>",
                  body, flags=re.S)
    # "Browse the eval logs" links become buttons
    body = re.sub(r'<p>(<a href="logs/[^"]+">[^<]*</a>)</p>', r'<p class="browse">\1</p>', body)
    return body


def main(serve: bool) -> None:
    text = SRC.read_text()
    m = re.match(r"\s*#\s+(.+)\n", text)
    title = m.group(1).strip() if m else "Write-up"
    text = text[m.end():] if m else text
    folders = sorted(set(re.findall(r"\(\.\./data/write-up/([\w.-]+)/?\)", text)))
    text = re.sub(r"\(\.\./data/write-up/([\w.-]+)/?\)", r"(logs/\1/)", text)
    images = sorted(set(re.findall(r"\]\((images/[^)\s]+)\)", text)))
    summary = re.search(r"## Executive summary\s+(.+?)\n", text)
    description = re.sub(r"[*_`\[\]]|\(\S+\)", "", summary.group(1))[:300] if summary else title

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    for img in images:
        shutil.copy(ROOT / "docs" / img, OUT / img)
    shutil.copy(SITE / "style.css", OUT / "style.css")
    for folder in folders:
        src = ROOT / "data" / "write-up" / folder
        if not src.is_dir():
            sys.exit(f"linked folder not found: {src}")
        subprocess.run(["uv", "run", "inspect", "view", "bundle", "--log-dir", str(src),
                        "--output-dir", str(OUT / "logs" / folder)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    (ROOT / "docs" / "images" / "valence_7b.svg").write_text(static_svg("valence"))   # for the README
    (OUT / ".nojekyll").write_text("")   # GitHub Pages: serve files as-is, no Jekyll processing

    body = figures_and_captions(to_html(text))
    page = (SITE / "template.html").read_text()
    for k, v in {"title": html.escape(title), "description": html.escape(description), "date": DATE,
                 "repo": REPO, "repo_short": REPO.removeprefix("https://"), "body": body}.items():
        page = page.replace("{{" + k + "}}", v)
    (OUT / "index.html").write_text(page)
    n_fig = body.count("<figure")
    print(f"built {OUT.relative_to(ROOT)} · {n_fig} figures/tables · {len(images)} image(s) · "
          f"{len(folders)} log viewer(s) · {datetime.datetime.now():%H:%M:%S}")

    if serve:
        subprocess.run([sys.executable, "-m", "http.server", "8000", "--directory", str(OUT)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    main(ap.parse_args().serve)
