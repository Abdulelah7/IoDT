"""Convert the IEEE Access manuscript (paper_rev/body_*.tex, appendix.tex) into the
Springer Nature sn-jnl template for Scientific Reports.  The front matter, the
IEEE-specific macros and the end statements are adapted to the journal format."""
import re, subprocess, pathlib, sys
H = pathlib.Path(__file__).resolve().parent
SRC = H.parent / "paper_rev"
TAB = H.parent / "Exp Data" / "tables"
sys.path.insert(0, str(H))
from bib_fix import apply as fix_bib

fill = lambda t: re.sub(r"%%T:(\w+)%%", lambda m: (TAB / f"{m.group(1)}.tex").read_text().rstrip("\n"), t)
body = "".join((SRC / f).read_text() for f in ["body_front.tex", "body_model.tex", "body_results.tex", "body_end.tex"])
app = (SRC / "appendix.tex").read_text()

# --- abstract and keywords from the IEEE front matter --------------------------
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()
body = body[body.index("\\maketitle") + len("\\maketitle"):]

# --- IEEE macros ------------------------------------------------------------------
body = body.replace("\\IEEEPARstart{T}{he}", "The")
def fig(m):
    width, path, cap, label = m.group(1), m.group(2), m.group(3).strip(), m.group(4)
    width = width.replace("\\columnwidth", "0.6\\textwidth")
    return ("\\begin{figure}[!t]\n\\centering\n\\includegraphics[" + width + "]{" + path + "}\n"
            "\\caption{" + cap + "}\\label{" + label + "}\n\\end{figure}")
body, nfig = re.subn(r"\\Figure\[[^\]]*\]\([^)]*\)\[([^\]]*)\]\{([^}]*)\}\s*\{(.*?)\\label\{([^}]*)\}\}", fig, body, flags=re.S)
assert nfig == 4, nfig

# --- end matter: Acknowledgment and Data/Code availability become statements -------
cut = body.index("\\section*{Acknowledgment}")
body = body[:cut].rstrip() + "\n"
app = ""  # appendix (proofs) removed at the author's request
body = body.replace("\n(Appendix~\\ref{app:proofs}).", ".")
body = re.sub(r"\\begin\{proof\}\s*See Appendix~\\ref\{app:proofs\}\.\s*\\end\{proof\}\n*", "\n", body)
assert "Appendix" not in body and "app:proofs" not in body
# --- Scientific Reports abstract (<= 200 words) ---
abstract = r"""The Internet of Digital Twins (IoDT) trust model pairs every IoT node
with a cloud twin and isolates a node when any of four binary
factors (energy, packet delivery ratio, traceability and twin
notification) fails.  This rule is fragile under congestion, trusts a
single twin, and keeps verdicts in a mutable database.  We propose
three extensions: an exponentially smoothed
weighted score with a per-factor persistence check; quorum voting over
signed votes of replicated twins; and a signed Merkle audit log kept
off the decision path.  In simulation at 4\% per-factor congestion,
the smoothed score falsely isolated 1.1\% of honest nodes within 16
rounds, against 93\% for the binary AND and 14--26\% for two-failure
rules, at the cost of one extra round.  On two public packet-delivery
traces it kept the lowest false isolation everywhere, although
bursty, concentrated real losses raised its rates to 2.0--31\%.
With identical evidence and two of five replicas corrupted, voting kept
detection at the level of an honest twin, whereas one corrupted
twin always hid its node.  The audit log rejected every edit by
adversaries holding fewer than a quorum of keys.  Without
congestion, an attacker that knows the rule can lie in 37.5\% of
rounds undetected."""

front = r"""\documentclass[lineno,pdflatex,sn-nature]{sn-jnl}

\usepackage{graphicx}%
\usepackage{multirow}%
\usepackage{amsmath,amssymb,amsfonts}%
\usepackage{amsthm}%
\usepackage{xcolor}%
\usepackage{textcomp}%
\usepackage{booktabs}%
\usepackage{array}%
\usepackage{float}%
\floatstyle{ruled}%
\newfloat{algorithm}{!t}{loa}%
\floatname{algorithm}{Algorithm}%
\floatstyle{plain}%

\theoremstyle{thmstyleone}%
\newtheorem{theorem}{Theorem}%
\newtheorem{proposition}{Proposition}%
\theoremstyle{thmstylethree}%
\newtheorem{definition}{Definition}%

\raggedbottom

\begin{document}

\title[Congestion-Tolerant and Auditable Trust Management for IoDT]{Congestion-Tolerant and Auditable Trust Management for the Internet of Digital Twins}

\author*{\fnm{Abdulelah} \sur{Alwabel}}\email{a.alwabel@psau.edu.sa}

\affil*{\orgdiv{Department of Computer Sciences, College of Computer Engineering and Sciences}, \orgname{Prince Sattam bin Abdulaziz University}, \orgaddress{\city{Al-Kharj}, \country{Saudi Arabia}}}

\abstract{""" + abstract + r"""}

\keywords{Audit log, Digital twin, Fog computing, Internet of Things, Majority voting, Trust management}

\maketitle
"""

back = r"""
\backmatter

\section*{Data availability}

The trace-driven experiments use the Intel Lab data~\cite{bodik2004intel}
and the FlockLab link-quality dataset~\cite{jacob2019flocklab}.
The raw results of every
experiment, with the configuration of every run, and the pre-processed
traces are available at \url{https://github.com/Abdulelah7/IoDT}.

\section*{Code availability}

The simulator, the experiment, table and figure scripts and the trace
pre-processing scripts are available in the same repository
(\url{https://github.com/Abdulelah7/IoDT}).

%%BIB%%

\bmhead{Author contributions}

A.A. conceived the study, designed the extensions, implemented the simulator,
performed the experiments and analysis, and wrote the manuscript.

\bmhead{Funding}

This work was supported by Prince Sattam bin Abdulaziz University under
Project PSAU/2026/R/1448.

\bmhead{Additional information}

\textbf{Competing interests:} The author declares no competing interests.

"""
tex = front + fill(body) + "\n" + fill(app) + back + "\\end{document}\n"
# layout only: labelled assumption lists ((A1), (B1), (C1)...) get a margin wide
# enough for their labels, so the labels do not run into the line numbers
def _lists(m):
    blk = m.group(0)
    if "\\item[(" not in blk:
        return blk
    return (blk.replace("\\begin{itemize}", "\\begin{list}{}{\\setlength{\\leftmargin}{3.4em}\\setlength{\\labelwidth}{3.0em}\\setlength{\\labelsep}{0.4em}\\setlength{\\itemsep}{1pt}\\setlength{\\parsep}{0pt}}", 1)
               .replace("\\end{itemize}", "\\end{list}"))
tex = re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", _lists, tex, flags=re.S)
# algorithm step lists: no blank gap after "Procedure:" / before bottom rule
def _alg(m):
    return (m.group(0)
            .replace("\\begin{enumerate}\\setlength{\\itemsep}{0pt}\\setlength{\\parskip}{0pt}",
                     "\\begin{list}{\\arabic{enumi}.}{\\usecounter{enumi}\\setlength{\\topsep}{1pt}\\setlength{\\partopsep}{0pt}\\setlength{\\itemsep}{0pt}\\setlength{\\parsep}{0pt}\\setlength{\\leftmargin}{1.6em}\\setlength{\\labelwidth}{1.2em}\\setlength{\\labelsep}{0.4em}}")
            .replace("\\end{enumerate}", "\\end{list}\\vspace{-2pt}"))
tex, nalg = re.subn(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", _alg, tex, flags=re.S)
assert nalg == 3, nalg
# layout only: fit the two wide tables to the single-column page
i = tex.index("\\label{tab:headline}")
tex = tex[:i] + tex[i:].replace("\\setlength{\\tabcolsep}{5pt}", "\\setlength{\\tabcolsep}{2.5pt}", 1)
i = tex.rindex("\\begin{table*}", 0, tex.index("\\label{tab:comparison}"))
j = tex.index("\\end{table*}", i)
tex = tex[:i] + tex[i:j].replace("\\begin{table*}[!t]", "\\begin{sidewaystable}") + "\\end{sidewaystable}" + tex[j + len("\\end{table*}"):]
run = lambda c: subprocess.run(c, cwd=H, capture_output=True, text=True).stdout
# --- wording adjustments ----------------------------------------------------------
FIXES = [
    ("at lower congestion it removes them.",
     "at $q=0.005$ it nearly removes them and at $q=0.001$ it removes them."),
    ("16 rounds, two significant figures;",
     "16 rounds; rates to two significant figures, naive-liar TTD to two decimals;"),
    ("labelled C1 and C2 in Fig.", "labelled M1 and M2 in Fig."),
    ("scripts are provided with the submission.",
     "scripts are publicly available (see Code availability)."),
    ("The tuning grids are provided with the\nsupplementary material.",
     "The tuning grids are included in the same repository."),
    ("the persistence check below, which keeps", "the persistence check above, which keeps"),
    ("\\subsection{Corrupted, Withholding and Slow Twins}",
     "\\subsection{Corrupted, Withholding and Slow Replicas}"),
    ("The mean time to the first false isolation (MTTFI) does not depend on",
     "The MTTFI does not depend on"),
    ("mean time to a first false isolation", "mean time to the first false isolation"),
    ("\\subsection{Limitations}", "\\subsection{Limitations}\n\\label{sec:limitations}"),
    ("Section~\\ref{sec:discussion}: a rule-aware", "Section~\\ref{sec:limitations}: a rule-aware"),
    ("with probability $q$ (assumption A1).", "with probability $q$ (assumptions (A1)--(A2))."),
    ("contract, so no ledger operation lies on the decision path.  The\nverdict and the signed votes are appended to the off-chain store of\nExtension~III.",
     "contract, so no ledger operation lies on the decision path.  The\nengine appends the signed votes to the same off-chain store."),
    ("May delete or rewrite; rewrites are rejected by the verifier (Theorem~\\ref{thm:tamper});",
     "May delete or rewrite; rewrites are rejected by the verifier unless made with $k$ replica keys before a newer checkpoint (Theorem~\\ref{thm:tamper});"),
    ("keys, and rewrite or delete records in the cloud.  Compromised nodes\nbehave in one of four ways.",
     "keys, and rewrite or delete records in the cloud.  The adversary acts\nin one of four ways."),
    ("The adversary controls $c$ replicas\nof the twins that watch the two compromised nodes.",
     "The adversary controls $c$ of the $n$ replicas\nof each twin, for compromised and honest nodes alike."),
    ("The congestion of the previous sections is independent across factors,",
     "The synthetic congestion of Sections~\\ref{sec:res-ext1}--\\ref{sec:res-ext3} is independent across factors,"),
    ("are detected slowly (Sec.~\\ref{sec:res-ext1}, \\ref{sec:res-stealth})",
     "are detected slowly (Sec.~\\ref{sec:res-stealth})"),
    ("($600$ runs per cell, attackers honest)", "($600$ runs per cell, no attackers)"),
    ("\\item The trust engine forms the header with\n", "\\item The trust engine forms the header with the Merkle tree hashes\n"),
    ("Keyless Signature Infrastructure of Buldas", "Keyless Signature Infrastructure (KSI) of Buldas"),
    ("unavailability, and every rule keeps its previous verdict.",
     "unavailability, and every rule keeps its previous verdict (the \\emph{silence rule})."),
    ("\\multicolumn{8}{l}{n.r.\\ = not reported.} \\\\",
     "\\multicolumn{8}{l}{n.r.\\ = not reported; n/a = not applicable (no blockchain).} \\\\"),
    ("false notifications and a denial-of-service attacker~", "false notifications and a denial-of-service (DoS) attacker~"),
    ("Internet of Things, Majority voting, Trust management}", "Internet of Things, Quorum voting, Trust management}"),
    ("computational overhead of blockchain validation is the reason Soula",
     "computational overhead of blockchain validation are the reasons Soula"),
    ("\\subsubsection{False-positive analysis}", "\\subsubsection{False-Positive Analysis}"),
    ("(trust set-up, block operations,", "(trust setup, block operations,"),
    ("& trust set-up &", "& trust setup &"),
]
for a, b in FIXES:
    assert tex.count(a) == 1, a
    tex = tex.replace(a, b)

# --- plain (non-bold) text in body and tables --------------------------------------
def strip_cmd(s, cmd):
    out, i = [], 0
    while True:
        j = s.find(cmd + "{", i)
        if j < 0:
            return "".join(out) + s[i:]
        out.append(s[i:j]); k = j + len(cmd) + 1; d = 1; st = k
        while d:
            d += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        out.append("{" + s[st:k - 1] + "}"); i = k
tex = strip_cmd(tex, "\\textbf")
assert "\\textbf" not in tex and "\\bfseries" not in tex

(H / "main.tex").write_text(tex.replace("%%BIB%%", "\\bibliography{refs}"))
run(["pdflatex", "-interaction=nonstopmode", "main.tex"]); run(["bibtex", "main"])
bbl = fix_bib((H / "main.bbl").read_text())
for a, b in [("\\bibinfo{author}{Lafuente, A.~L.}", "\\bibinfo{author}{Lluch Lafuente, A.}"),
             ("Electronic Commerce Conf.} paper 41 (2002).",
              "Electronic Commerce Conf.} paper 41 (AIS eLibrary, 2002). \\url{https://aisel.aisnet.org/bled2002/41}."),
             ("\\url{https://db.csail.mit.edu/labdata/labdata.html} (2004).",
              "\\url{https://db.csail.mit.edu/labdata/labdata.html} (2004; accessed 12 October 2026).")]:
    assert bbl.count(a) == 1, a
    bbl = bbl.replace(a, b)
(H / "main.tex").write_text(tex.replace("%%BIB%%", bbl))
for _ in range(2):
    log = run(["pdflatex", "-interaction=nonstopmode", "main.tex"])
print(re.findall(r"Output written.*", log))
