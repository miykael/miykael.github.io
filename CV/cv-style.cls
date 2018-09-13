% -- Encoding UTF-8
\ProvidesClass{cv-style}[2018/07/31 CV class]
\NeedsTeXFormat{LaTeX2e}

\DeclareOption{english}{\def\@cv@english{}}
\DeclareOption{print}{\def\@cv@print{}}
\DeclareOption*{
  \PassOptionsToClass{\CurrentOption}{article}
}
\ProcessOptions\relax
\LoadClass{article}

%----------------------------------------------------------------------------------------
% Colors %
%----------------------------------------------------------------------------------------

\RequirePackage{xcolor}

\definecolor{white}{RGB}{255,255,255}

\definecolor{darkgray}{HTML}{333333}
\definecolor{gray}{HTML}{4D4D4D}
\definecolor{lightgray}{HTML}{999999}

\definecolor{date}{HTML}{A4A4A4}

\ifdefined\@cv@print
  \colorlet{fillheader}{white}
  \colorlet{header}{gray}
\else
  \colorlet{fillheader}{gray}
  \colorlet{header}{white}
\fi
\colorlet{textcolor}{gray}
\colorlet{headercolor}{gray}

%----------------------------------------------------------------------------------------
% Fonts %
%----------------------------------------------------------------------------------------

\RequirePackage[quiet]{fontspec}
\RequirePackage[math-style=TeX,vargreek-shape=unicode]{unicode-math}

\newfontfamily\bodyfont{Roboto-Regular}[Path=fonts/]
\newfontfamily\bodyfontit{Roboto-LightItalic}[Path=fonts/]
\newfontfamily\thinfont{Roboto-Thin}[Path=fonts/]
\newfontfamily\headingfont{RobotoCondensed-Bold}[Path=fonts/]

\defaultfontfeatures{Mapping=tex-text}
\setmainfont[Mapping=tex-text, Color=textcolor, Path = fonts/]{Roboto-Light}

\newcommand{\italica}[1]{%
    {\color{gray}\bodyfontit #1}%
}

%----------------------------------------------------------------------------------------
% Header %
%----------------------------------------------------------------------------------------

\RequirePackage{tikz}

\newcommand{\header}[2]{%
  \begin{tikzpicture}[remember picture,overlay]
    \node [rectangle, fill=fillheader, anchor=north, minimum width=\paperwidth, minimum height=1.5cm] (box) at (current page.north){};
    \node [anchor=center] (name) at (box) {%
      \fontsize{32pt}{65pt}\color{header}%
      {\thinfont #1}{\bodyfont  #2}
    };
  \end{tikzpicture}
  \vspace{0.25cm}
  \vspace{-1\parskip}
}

%----------------------------------------------------------------------------------------
% Last updated command %
%----------------------------------------------------------------------------------------

\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{polyglossia}

\newcommand{\sethyphenation}[3][]{%
  \sbox0{\begin{otherlanguage}[#1]{#2}
    \hyphenation{#3}\end{otherlanguage}}}

\ifdefined\@cv@english
  \setdefaultlanguage{english}
  \def\lastupdatedtext{}
\else
  \setdefaultlanguage[variant=british]{english}
  \def\lastupdatedtext{Last Updated on}
\fi

\setlength{\TPHorizModule}{0.01\paperwidth}
\setlength{\TPVertModule}{0.01\paperwidth}

\newcommand{\lastupdated}{
  \begin{textblock}{10}(11.3, 0.05)
    \raggedleft
    \fontsize{8pt}{10pt}\color{date}\thinfont
    \lastupdatedtext{} \today
  \end{textblock}}

%----------------------------------------------------------------------------------------
% Structure %
%----------------------------------------------------------------------------------------
\RequirePackage{parskip}

\newcounter{colorCounter}
\def\@sectioncolor#1#2#3{%
  {%
    \color{%
      \ifcase\value{colorCounter}%
        gray\else%
        headercolor\fi%
    } #1#2#3%
  }%
  \stepcounter{colorCounter}%
}

\renewcommand{\section}[1]{
  {\par\vspace{\parskip}
  {%
    \LARGE\headingfont\color{headercolor}%
    \@sectioncolor #1%
  }
  \par\vspace{\parskip}}
}

\renewcommand{\subsection}[2]{
  \par\vspace{.5\parskip}%
  \Large\headingfont\color{headercolor} #2%
  \par\vspace{.25\parskip}%
}

\newcommand{\jobtitle}[1]{%
    {\color{gray}\bodyfontit #1}%
}

\pagestyle{empty}

%----------------------------------------------------------------------------------------
% List environment %
%----------------------------------------------------------------------------------------

\setlength{\tabcolsep}{0pt}
\newenvironment{entrylist}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}
\renewcommand{\bfseries}{\headingfont\color{headercolor}}
\newcommand{\entry}[4]{%
  #1&\parbox[t]{12.2cm}{%
    \textbf{#2}%
    \hfill%
    {\footnotesize\addfontfeature{Color=grey} #3}\\%
    #4\vspace{\parsep}%
  }\\}

%----------------------------------------------------------------------------------------
% Professional Activities environment %
%----------------------------------------------------------------------------------------

\setlength{\tabcolsep}{0pt}
\newenvironment{talklist}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}
\renewcommand{\bfseries}{\headingfont\color{headercolor}}
\newcommand{\talks}[4]{%
  #1&\parbox[t]{18.25cm}{%
    \large\textbf{#2}%
    \normalsize
    \hfill%
    {\footnotesize\addfontfeature{Color=grey} #3}\\%
    #4\vspace{\parsep}%
  }\\}

%----------------------------------------------------------------------------------------
% Publications environment %
%----------------------------------------------------------------------------------------

\setlength{\tabcolsep}{0pt}
\newenvironment{publist}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}
\renewcommand{\bfseries}{\headingfont\color{headercolor}}
\newcommand{\publications}[2]{%
  #1&\parbox[t]{18.25cm}{%
    #2\vspace{\parsep}%
  }\\}

%----------------------------------------------------------------------------------------
% Reference environment %
%----------------------------------------------------------------------------------------

\setlength{\tabcolsep}{0pt}
\newenvironment{reflist}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}
\renewcommand{\bfseries}{\headingfont\color{headercolor}}
\newcommand{\reference}[2]{%
  #1&\parbox[t]{16.5cm}{%
    #2\vspace{\parsep}%
  }\\}

%----------------------------------------------------------------------------------------
% Side block %
%----------------------------------------------------------------------------------------

\setlength{\TPHorizModule}{1cm}
\setlength{\TPVertModule}{0.70cm}
\newenvironment{aside}{%
  \let\oldsection\section
  \renewcommand{\section}[1]{
    \par\vspace{\baselineskip}{\Large\headingfont\color{headercolor} ##1}
  }
  \begin{textblock}{4.6}(1, 1.87)
  \begin{flushleft}
  \obeycr
}{%
  \restorecr
  \end{flushleft}
  \end{textblock}
  \let\section\oldsection
}

%----------------------------------------------------------------------------------------
% Other tweaks %
%----------------------------------------------------------------------------------------

\RequirePackage[left=5.8cm,top=1cm,right=1cm,bottom=1cm,nohead,nofoot]{geometry}
\RequirePackage[colorlinks=true, linkcolor=blue, urlcolor=blue]{hyperref}
\hypersetup{
    pdftitle=CV/Resume \textbar{} Michael P. Notter,
    pdfauthor=Michael P. Notter,
    pdfsubject=CV/Resume%
    }
