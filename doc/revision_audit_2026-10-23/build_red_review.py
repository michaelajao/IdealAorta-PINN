"""Render the retained comparison with revised text red and deletions hidden.

Run from the repository root. The canonical manuscript remains main.tex.
The supplied comparison source is retained verbatim as evidence.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
source = (HERE / 'codex_changes_vs_abaid.tex').read_text(encoding='utf-8')
overrides = r'''
% Coauthor review: current wording only; every diff addition is red.
\renewcommand{\DIFaddtex}[1]{{\protect\color{red}#1}}
\renewcommand{\DIFdeltex}[1]{}
\renewcommand{\DIFdel}[1]{}
\renewcommand{\DIFdelFL}[1]{}
\renewcommand{\DIFaddincludegraphics}[2][]{\DIFOincludegraphics[#1]{#2}}
\renewcommand{\DIFdelincludegraphics}[2][]{}
\renewcommand{\cfdrevision}[1]{\textcolor{red}{#1}}
\renewcommand\thesection{\arabic{section}}
\renewcommand\thesubsection{\thesection.\arabic{subsection}}
\pdfstringdefDisableCommands{%
  \def\DIFaddbegin{}\def\DIFaddend{}%
  \def\DIFdelbegin{}\def\DIFdelend{}%
  \def\DIFaddbeginFL{}\def\DIFaddendFL{}%
  \def\DIFdelbeginFL{}\def\DIFdelendFL{}%
  \def\cfdrevision#1{#1}%
}
'''
source = source.replace(r'\begin{document}', overrides + '\n' + r'\begin{document}', 1)
# Diff macros can hide the next word from acronym xspace handling.
source = re.sub(r'(\\(?:CFD|WSS|TAWSS|RBF|PINNs|PINN|DTAA|TKE|MPS|DNS|FSI))\s+(?=\\DIFadd\{[A-Za-z])',
                lambda match: match.group(1) + '{}\\ ', source)
# These captions are separate input files, so the supplied main-file diff
# cannot mark their edits. Mark the current captions in this review only.
for name in ('reference_floor', 'robustness'):
    table = (ROOT / 'doc/IdealAorta_EACFM/tables' / (name + '.tex')).read_text(encoding='utf-8')
    start = table.index(r'\caption{') + len(r'\caption{')
    level, end = 1, start
    while level:
        if table[end] == '{' and table[end-1] != '\\': level += 1
        if table[end] == '}' and table[end-1] != '\\': level -= 1
        end += 1
    table = table[:start] + r'\textcolor{red}{' + table[start:end-1] + '}' + table[end-1:]
    source = source.replace(r'\input{tables/' + name + '}', table)
target = HERE / 'main_red_review.tex'
target.write_text(source, encoding='utf-8', newline='\n')
output = ROOT / 'report/red_review_build'
output.mkdir(parents=True, exist_ok=True)
with (output / 'build_console.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                             '-outdir=' + str(output), str(target)],
                            cwd=ROOT / 'doc/IdealAorta_EACFM', stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    raise SystemExit('Build failed; see report/red_review_build/build_console.log')
print(output / 'main_red_review.pdf')
