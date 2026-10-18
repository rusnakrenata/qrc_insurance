"""Write main_flat.tex: main.tex with every \\input file and the bibliography (main.bbl) inlined,
and pack it with the figures and IEEEtran class into main_tex_submission.zip.

  cd article && python flatten.py      (run after pdflatex/bibtex so that main.bbl exists)
"""
import os
import re
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def inline(text):
    def rep(m):
        name = m.group(1)
        path = os.path.join(HERE, name if name.endswith(".tex") else name + ".tex")
        return "%% ---- " + name + "\n" + inline(open(path).read()).rstrip() + "\n"
    return re.sub(r"\\input\{([^}]+)\}", rep, text)


def main():
    tex = inline(open(os.path.join(HERE, "main.tex")).read())
    bbl = open(os.path.join(HERE, "main.bbl")).read()
    tex = tex.replace("\\graphicspath{{../figures/}}", "\\graphicspath{{./}}").replace("\\graphicspath{{./}{../figures/}}", "\\graphicspath{{./}}")
    tex = re.sub(r"\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}", lambda m: bbl, tex)
    with open(os.path.join(HERE, "main_flat.tex"), "w") as f:
        f.write(tex)
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)))
    with zipfile.ZipFile(os.path.join(HERE, "main_tex_submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(HERE, "main_flat.tex"), "main.tex")
        z.write(os.path.join(HERE, "IEEEtran.cls"), "IEEEtran.cls")
        for f in figs:
            for ext in ("", ".pdf", ".png"):
                p = os.path.join(HERE, "..", "figures", f + ext)
                if os.path.isfile(p):
                    z.write(p, f + ext)
                    break
            else:
                raise FileNotFoundError(f)
    print("main_flat.tex and main_tex_submission.zip written;", len(figs), "figures")


if __name__ == "__main__":
    main()
