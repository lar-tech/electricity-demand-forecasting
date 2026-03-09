\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{setup/setup}[2025/08/13 v0.1 KOMA article class]

% options
\RequirePackage{kvoptions}
\SetupKeyvalOptions{family=setup,prefix=setup@}
\DeclareStringOption[setup/references.bib]{bibresource}
\DeclareBoolOption[true]{listings}
\ProcessKeyvalOptions*

% base-class
\LoadClass[a4paper,oneside,12pt,titlepage]{scrartcl}

% packages
\RequirePackage[hidelinks]{hyperref}
\RequirePackage[printonlyused]{acronym}
\RequirePackage[english]{babel}
\usepackage[ngerman]{datetime}
\newdateformat{myformat}{\THEDAY{.\ }\monthname[\THEMONTH] \THEYEAR}
\newdateformat{americandateformat}{\monthname[\THEMONTH] \THEDAY{, }\THEYEAR}

% maths and units
\RequirePackage{trfsigns}
\RequirePackage{mathrsfs}
\RequirePackage{amsfonts,amsmath,amssymb}
\allowdisplaybreaks
\RequirePackage{siunitx}
\sisetup{locale=DE,output-decimal-marker={,}}

% figures and tables
\RequirePackage{chngcntr}
\RequirePackage{array}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\RequirePackage{graphicx}
\RequirePackage[export]{adjustbox}
\RequirePackage{multirow,multicol}
\RequirePackage{subcaption}
\RequirePackage{booktabs}
\RequirePackage{textcomp}
\RequirePackage{float}
\RequirePackage[section]{placeins}
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}
\RequirePackage[dvipsnames]{xcolor}

% listings
\ifsetup@listings
  \RequirePackage{matlab-prettifier}
  \RequirePackage{listings}
  \lstset{
    frame=single,
    numbers=left,
    style=Matlab-editor
  }
\fi

% header/footer
\RequirePackage{scrlayer-scrpage}
\setkomafont{pagehead}{\normalfont}
\setkomafont{pagefoot}{\normalfont}
\setkomafont{pagination}{\normalfont}
\KOMAoptions{headsepline=0.5pt}
\KOMAoptions{footsepline=0.5pt}

% spacings
\voffset26pt
\parskip6pt
\RequirePackage{setspace}
\onehalfspacing
\setlength{\parindent}{0pt}
\overfullrule 5pt

% notes
\setlength{\marginparwidth}{2cm}
\RequirePackage[colorinlistoftodos]{todonotes}

% counter
\setcounter{secnumdepth}{5}
\setcounter{tocdepth}{5}
\renewcommand{\lstlistingname}{Code}
\AtBeginDocument{%
  \numberwithin{equation}{section}%
  \numberwithin{lstlisting}{section}%
  \counterwithin{figure}{section}%
  \counterwithin{table}{section}%
}

% tikz, pgfplots
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta, decorations.pathreplacing, calc,3d,external}
\RequirePackage{pgfplots}
\usepgfplotslibrary{dateplot, groupplots, colormaps, fillbetween}
\pgfplotsset{
    compat=1.9,
    small axis/.style={
        legend style={
            at={(1,1)},
            anchor=north east,
            xshift=-3pt,
            yshift=-3pt,
            font=\scriptsize\selectfont,
            rounded corners=2pt,
            fill=white,
            fill opacity=0.7,
            text opacity=1,
            draw=black!20,
            row sep=0pt,         
            legend image code/.code={\draw plot coordinates {(0cm,0cm) (0.3cm,0cm)};},
        },
        label style={font=\scriptsize},
        scaled ticks=false,
        /pgf/number format/.cd,
            % assume math mode=true,
            fixed,
            precision=0,
            1000 sep={},
        tick label style={font=\sffamily\scriptsize},
    },
    colormap={shap}{
        rgb255=(0,138,250)
        rgb255=(0,124,245)
        rgb255=(35,108,234)
        rgb255=(97,89,218)
        rgb255=(132,66,197)
        rgb255=(155,34,172)
        rgb255=(186,0,159)
        rgb255=(213,0,142)
        rgb255=(233,0,123)
        rgb255=(248,0,102)
        rgb255=(255,0,81)
    }
}
\newlength{\imageheight}
\setlength{\imageheight}{6cm}
\definecolor{tured}{RGB}{180,41,42}
\definecolor{tabblue}{rgb}{0.1216, 0.4667, 0.7059}
\definecolor{taborange}{rgb}{1.0000, 0.4980, 0.0549}
\definecolor{tabgreen}{rgb}{0.1725, 0.6275, 0.1725}
\definecolor{tabred}{rgb}{0.8392, 0.1529, 0.1569}
\definecolor{tabpurple}{rgb}{0.5804, 0.4039, 0.7412}

% title 
\newcommand{\department}[1]{\def\@department{#1}}
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\def\@authorlist{}
\newcommand{\addauthor}[4][-1cm]{\listadd{\@authorlist}{{#2}{#3}{#4}{#1}}}
\newcommand{\@formatsignature}[1]{\@formatsignatureinner#1}
\newcommand{\@formatsignatureinner}[4]{%
    \makebox[5cm][c]{\includegraphics[width=0.2\linewidth]{#3}}\\[#4]
    \rule{5cm}{0.5pt} \\[0.2cm]
    \textsc{#1} \\[0.8cm]
}
\newcommand{\@formatauthor}[1]{\@formatauthorinner#1}
\newcommand{\@formatauthorinner}[4]{& #1 & (#2)\\[1.2ex]}
\newcommand{\maketitlepage}{%
    \thispagestyle{plain}
    \begin{titlepage}
        \begin{center}
            \includegraphics[width=6cm, valign=c]{setup/figures/logo_tub.pdf}%
            \hfill
            \includegraphics[width=3cm, valign=c]{setup/figures/logo_control_systems.pdf}
        \end{center}
        \vspace{2cm}
        \begin{center}
            \LARGE{\textbf{\@title}}\\[1ex]
            \vspace{1.0cm}
            \normalsize
            \vspace{0.25cm}
            at the Department of \\
            \vspace{0.25cm}
            \textbf{\@department}\\
            \vspace{0.25cm}
            \textbf{Faculty IV -- Electrical Engineering and Computer Science \\
            Institute for Energy and Automation Technology}
            \vspace{1cm}
            \begin{table}[H]
                \centering
                \begin{tabular}{lll}
                    Authors: \forlistloop{\@formatauthor}{\@authorlist}
                    Supervisor: & \@supervisor \\[1.2ex]
                    Submission date: & \americandateformat\@date \\[3ex]
                \end{tabular}
            \end{table}
        \end{center}
        \vspace*{\fill}
    \end{titlepage}
}

% statutory declaration
\newcommand{\makedeclaration}{%
    \begin{otherlanguage*}{ngerman}
    \section*{Eigenständigkeitserklärung}
    Hiermit versichere ich, dass ich die vorliegende Arbeit eigenständig ohne Hilfe Dritter und ausschließlich unter Verwendung der aufgeführten Quellen und Hilfsmittel angefertigt habe.Alle Stellen, die den benutzten Quellen und Hilfsmitteln unverändert oder sinngemäß entnommen sind, habe ich als solche kenntlich gemacht.
    
    Sofern generische KI-Tools verwendet wurden, habe ich Produktnamen, Hersteller, die jeweils verwendete Softwareversion und die jeweiligen Einsatzzwecke (z.B. sprachliche Überprüfung und Verbesserung der Texte, systematische Recherche) benannt. Ich verantworte die Auswahl, die Übernahme und sämtliche Ergebnisse des von mir verwendeten KI-generierten Outputs vollumfänglich selbst.
    
    Ich erkläre weiterhin, dass ich die Arbeit in gleicher oder ähnlicher Form noch keiner anderen Prüfungsbehörde vorgelegt habe.

    Die \href{https://www.static.tu.berlin/fileadmin/www/10002457/K3-AMBl/Amtsblatt_2023/Amtliches_Mitteilungsblatt_Nr._16_vom_30.05.2023.pdf}{Satzung zur Sicherung guter wissenschaftlicher Praxis an der TU Berlin} vom 15. Februar 2023 habe ich zur Kenntnis genommen.

    \vspace{0.5cm}
    Berlin, den \myformat\@date

    \vspace{0.5cm}
    \begin{tabular}{@{}l@{}}
        \forlistloop{\@formatsignature}{\@authorlist}
    \end{tabular}
    \end{otherlanguage*}
    \clearpage
}

% bibliography
\RequirePackage{csquotes}
\RequirePackage[style=ieee,doi=true,isbn=false,url=false,mincitenames=1,maxcitenames=3,minbibnames=10,maxbibnames=10]{biblatex}
\DeclareFieldFormat{title}{#1}
\DeclareFieldFormat [article,book,inbook,incollection,proceedings,inproceedings,report,thesis,online,misc,software]{title}{\mkbibemph{#1}}
\DeclareFieldFormat{note}{(#1)}
\DeclareFieldFormat{sentencecase}{#1}
\RequirePackage{xpatch}
\xpatchbibmacro{textcite}{\addspace}{\addnbspace}{}{}
\xpatchbibmacro{Textcite}{\addspace}{\addnbspace}{}{}
\DefineBibliographyStrings{english}{andothers = et~al\adddot\addspace}
\defbibheading{References}[\bibname]{%
	\section*{#1}%
	\addcontentsline{toc}{section}{#1}%
	\sectionmark{#1}%
}
\bibliography{setup/references}