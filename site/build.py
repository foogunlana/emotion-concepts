"""Compile the write-up into the GitHub Pages site.

    uv run --with markdown python site/build.py            # build into site/_build/
    uv run --with markdown python site/build.py --serve    # build, then serve on http://127.0.0.1:8000

Pipeline:
  docs/write-up-2.md  --markdown-->  HTML body
    · the first "# heading" becomes the page title
    · an image or table followed by a paragraph that is all italics becomes a <figure> with that caption
    · a paragraph containing only {{figure:NAME}} is replaced by the inline SVG from site/figures.py
    · links to ../data/write-up/<folder>/[#/tasks/…] point to a static Inspect viewer bundled at logs/<folder>/
  site/template.html + site/style.css wrap the body; images are copied from docs/images/.
"""
import argparse, datetime, html, re, shutil, subprocess, sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import FIGURES, static_svg, transcript

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SRC = ROOT / "docs" / "write-up-2.md"
OUT = SITE / "_build"
REPO = "https://github.com/foogunlana/emotion-concepts"
DATE = "September 2026"
AUTHOR = "Foluso Ogunlana"
ICONS = {   # brand marks, as inline SVG
    "Hugging Face": '<span class="hf-icon" aria-hidden="true">🤗</span>',
    "GitHub": '<svg viewBox="0 0 16 16" width="20" height="20" aria-hidden="true"><path fill="#1a1a18" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>',
    "LinkedIn": '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path fill="#0a66c2" d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>',
}
MENTOR = {"name": "Alexander Reinthal", "photo": "images/mentor.png",
          "links": {}}
AUTHOR_LINKS = {"GitHub": "https://github.com/foogunlana",
                "LinkedIn": "https://www.linkedin.com/in/foluso-olabode-bode-ogunlana-124522a4/",
                "Hugging Face": "https://huggingface.co/foogunlana"}


def to_html(md_text: str) -> tuple[str, list]:
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"],
                           extension_configs={"toc": {"permalink": False}})
    return md.convert(md_text), md.toc_tokens


def toc_html(tokens: list) -> str:
    """Contents sidebar: every ## section, with its ### subsections nested."""
    def item(t, level):
        kids = "".join(item(c, level + 1) for c in t.get("children", [])) if level < 3 else ""
        sub = f'<ol>{kids}</ol>' if kids else ""
        return f'<li class="l{level}"><a href="#{t["id"]}">{t["name"]}</a>{sub}</li>'
    return "".join(item(t, 2) for t in tokens)


def icon_links(links: dict) -> str:
    return "".join(f'<a class="icon-link" href="{u}" title="{k}" aria-label="{k}">{ICONS[k]}</a>' for k, u in links.items())


def parse_fields(block: str) -> dict:
    """`key: value` lines; a line without a key continues the previous value."""
    fields, key = {}, None
    for line in block.splitlines():
        m = re.match(r"(\w+):\s?(.*)", line)
        if m and m.group(1) in {"id", "label", "kind", "summary", "user", "assistant", "note", "link", "open"}:
            key = m.group(1); fields[key] = m.group(2)
        elif key:
            fields[key] += "\n" + line
    return fields


def inline_md(text: str) -> str:
    html_ = markdown.markdown(text.strip())
    return re.sub(r"^<p>|</p>$", "", html_)


def figures_and_captions(body: str) -> str:
    caption = r'\s*<p><em>((?:(?!</p>).)*?)</em>((?:(?!</p>).)*?)</p>'
    # chart placeholders (+ optional italic caption)
    def chart(m):
        svg = FIGURES[m.group(1)]()
        cap = f"<figcaption>{m.group(2)}{m.group(3) or ''}</figcaption>" if m.group(2) else ""
        return f'<figure class="chart fig-{m.group(1)}">{svg}{cap}</figure>'
    body = re.sub(r"<p>\{\{figure:([\w-]+)\}\}</p>(?:" + caption + ")?", chart, body, flags=re.S)
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
    link = r"\(\.\./data/write-up/([\w.-]+)/?(#[^)\s]*)?\)"      # optional #fragment = a sample deep link
    folders = sorted(set(m.group(1) for m in re.finditer(link, text)))
    text = re.sub(link, lambda m: f"(logs/{m.group(1)}/{m.group(2) or ''})", text)
    images = sorted(set(re.findall(r"\]\((images/[^)\s]+)\)", text)))
    text = re.sub(r"```transcript\n(.*?)\n```", lambda m: "\n" + transcript(parse_fields(m.group(1)), inline_md) + "\n",
                  text, flags=re.S)
    thanks = re.search(r"^## Thanks\s*\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    thanks_html = markdown.markdown(thanks.group(1).strip()) if thanks else ""
    if thanks:
        text = text[:thanks.start()] + text[thanks.end():]
    summary = re.search(r"## Executive summary\s+(.+?)\n", text)
    description = re.sub(r"[*_`\[\]]|\(\S+\)", "", summary.group(1))[:300] if summary else title

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    for img in images:
        shutil.copy(ROOT / "docs" / img, OUT / img)
    shutil.copy(SITE / "style.css", OUT / "style.css")
    for img in ("author.png", "mentor.png"):
        shutil.copy(ROOT / "docs" / "images" / img, OUT / "images" / img)
    for folder in folders:
        src = ROOT / "data" / "write-up" / folder
        if not src.is_dir():
            sys.exit(f"linked folder not found: {src}")
        subprocess.run(["uv", "run", "inspect", "view", "bundle", "--log-dir", str(src),
                        "--output-dir", str(OUT / "logs" / folder)], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
    (ROOT / "docs" / "images" / "valence_7b.svg").write_text(static_svg("valence"))   # for the README
    (OUT / ".nojekyll").write_text("")   # GitHub Pages: serve files as-is, no Jekyll processing

    body, tokens = to_html(text)
    body = figures_and_captions(body)
    page = (SITE / "template.html").read_text()
    for k, v in {"title": html.escape(title), "description": html.escape(description), "date": DATE,
                 "repo": REPO, "repo_short": REPO.removeprefix("https://"), "body": body,
                 "toc": toc_html(tokens), "author": html.escape(AUTHOR),
                 "author_links": icon_links(AUTHOR_LINKS), "mentor": html.escape(MENTOR["name"]),
                 "mentor_photo": MENTOR["photo"], "mentor_links": icon_links(MENTOR["links"]),
                 "thanks": thanks_html}.items():
        page = page.replace("{{" + k + "}}", v)
    import hashlib
    ver = hashlib.sha1((SITE / "style.css").read_bytes()).hexdigest()[:8]
    page = page.replace('href="style.css"', f'href="style.css?v={ver}"')   # bust browser caches on every change
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
