"""Nature-style replacements for entry types that sn-nature.bst renders badly
(conference papers, book chapters, datasets); applied to the .bbl by build.py."""
import re
FIX = {
"bao2012dynamic": r"Bao, F. \& Chen, I.-R. Dynamic trust management for {Internet} of {Things} applications. in \emph{Proc. 2012 Int. Workshop on Self-Aware Internet of Things} 1--6 (ACM, 2012).",
"sun2006trust": r"Sun, Y.~L., Han, Z., Yu, W. \& Liu, K. J.~R. A trust evaluation framework in distributed networks: vulnerability analysis and defense against attacks. in \emph{Proc. IEEE INFOCOM 2006} 1--13 (IEEE, 2006).",
"dorri2017blockchain": r"Dorri, A., Kanhere, S.~S., Jurdak, R. \& Gauravaram, P. Blockchain for {IoT} security and privacy: the case study of a smart home. in \emph{Proc. IEEE Int. Conf. Pervasive Computing and Communications Workshops (PerCom Workshops)} 618--623 (IEEE, 2017).",
"sleuters2019digital": r"Sleuters, J., Li, Y., Verriet, J., Velikova, M. \& Doornbos, R. A digital twin method for automated behavior analysis of large-scale distributed {IoT} systems. in \emph{Proc. 14th Annual Conf. System of Systems Engineering (SoSE)} 7--12 (IEEE, 2019).",
"empl2022soar4iot": r"Empl, P., Schlette, D., Zupfer, D. \& Pernul, G. {SOAR4IoT}: securing {IoT} assets with digital twins. in \emph{Proc. 17th Int. Conf. Availability, Reliability and Security (ARES)} 1--10 (ACM, 2022).",
"kamvar2003eigentrust": r"Kamvar, S.~D., Schlosser, M.~T. \& Garcia-Molina, H. The {EigenTrust} algorithm for reputation management in {P2P} networks. in \emph{Proc. 12th Int. Conf. World Wide Web (WWW)} 640--651 (ACM, 2003).",
"josang2002beta": r"J{\o}sang, A. \& Ismail, R. The beta reputation system. in \emph{Proc. 15th Bled Electronic Commerce Conf.} paper 41 (2002).",
"bonomi2012fog": r"Bonomi, F., Milito, R., Zhu, J. \& Addepalli, S. Fog computing and its role in the {Internet of Things}. in \emph{Proc. First MCC Workshop on Mobile Cloud Computing} 13--16 (ACM, 2012).",
"crosby2009efficient": r"Crosby, S.~A. \& Wallach, D.~S. Efficient data structures for tamper-evident logging. in \emph{Proc. 18th USENIX Security Symposium} 317--334 (USENIX Association, 2009).",
"kalodner2018arbitrum": r"Kalodner, H., Goldfeder, S., Chen, X., Weinberg, S.~M. \& Felten, E.~W. Arbitrum: scalable, private smart contracts. in \emph{Proc. 27th USENIX Security Symposium} 1353--1370 (USENIX Association, 2018).",
"jacob2019flocklab": r"Jacob, R., Da~Forno, R., Tr{\"u}b, R., Biri, A. \& Thiele, L. Dataset: wireless link quality estimation on {FlockLab} -- and beyond. in \emph{Proc. 2nd Workshop on Data Acquisition To Analysis (DATA)} 57--60 (ACM, 2019).",
"lim2013flocklab": r"Lim, R. \emph{et~al.} {FlockLab}: a testbed for distributed, synchronized tracing and profiling of wireless embedded systems. in \emph{Proc. 12th Int. Conf. Information Processing in Sensor Networks (IPSN)} 153--166 (ACM, 2013).",
"buldas2013ksi": r"Buldas, A., Kroonmaa, A. \& Laanoja, R. Keyless signatures' infrastructure: how to build global distributed hash-trees. in \emph{Secure IT Systems (NordSec 2013)}, Lecture Notes in Computer Science Vol.~8208, 313--320 (Springer, 2013).",
"douceur2002sybil": r"Douceur, J.~R. The {Sybil} attack. in \emph{Peer-to-Peer Systems (IPTPS 2002)}, Lecture Notes in Computer Science Vol.~2429, 251--260 (Springer, 2002).",
"laurie2013rfc6962": r"Laurie, B., Langley, A. \& Kasper, E. Certificate Transparency. RFC 6962 (RFC Editor, 2013). \url{https://doi.org/10.17487/RFC6962}.",
"malkhi1998byzantine": r"Malkhi, D. \& Reiter, M.~K. {Byzantine} quorum systems. \emph{Distrib. Comput.} \textbf{11}, 203--213 (1998).",
"bodik2004intel": r"Bodik, P. \emph{et~al.} Intel Lab data. Intel Berkeley Research Lab \url{https://db.csail.mit.edu/labdata/labdata.html} (2004).",
"jacob2019flocklabdata": r"Jacob, R., Da~Forno, R., Tr{\"u}b, R., Biri, A. \& Thiele, L. Wireless link quality estimation on {FlockLab} -- and beyond, version 1.4.1. \emph{Zenodo} \url{https://doi.org/10.5281/zenodo.3731498} (2020).",
"aws2024az": r"Amazon Web Services. Availability Zones. in \emph{AWS Fault Isolation Boundaries} (AWS whitepaper) \url{https://docs.aws.amazon.com/whitepapers/latest/aws-fault-isolation-boundaries/availability-zones.html} (accessed 12 October 2026).",
}
def apply(bbl):
    parts = re.split(r"(?=\\bibitem\{)", bbl)
    out = []
    for p in parts:
        m = re.match(r"\\bibitem\{([^}]+)\}", p)
        if m and m.group(1) in FIX:
            tail = "\n\n\\end{thebibliography}\n" if "\\end{thebibliography}" in p else "\n\n"
            p = f"\\bibitem{{{m.group(1)}}}\n{FIX[m.group(1)]}" + tail
        out.append(p)
    return "".join(out)
