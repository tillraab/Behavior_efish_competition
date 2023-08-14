\newcommand\classname{COB}
\newcommand\lastmodifieddate{2015/01/09}
\newcommand\versionnumber{0.1}

\newif\if@cropmarkson \@cropmarksontrue

\NeedsTeXFormat{LaTeX2e}[2000/06/01]
\ProvidesClass{\classname}[\lastmodifieddate\space (C) 2015 Techset Composition Pvt. Ltd.^^J^^J]

\special{papersize=8.5in,11in}

\setlength{\paperheight}{11in}%
\setlength{\paperwidth}{8.5in}%

\newif\if@numberedref \@numberedreftrue
\newif\if@endproofbox \@endproofboxtrue
\newif\ifpreview
\newif\ifrevised
\newif\ifopenacc
\newif\ifnostrip
\newif\ifonecolumn\global\onecolumnfalse%
\newif\ifnote@end\note@endfalse
\newif\ifxipt
\newif\ifnumbib
\newif\if@Vruler%
\newif\ifDEV
\newif\ifJEB
\newif\ifJCS
\newif\ifDMM
\newif\ifBIO


\DeclareOption{DEV}{\DEVtrue}
\DeclareOption{JEB}{\JEBtrue}
\DeclareOption{JCS}{\JCStrue}
\DeclareOption{DMM}{\DMMtrue}
\DeclareOption{BIO}{\BIOtrue}
\DeclareOption{numbib}{\numbibtrue}
\DeclareOption{xipt}{\xipttrue}
\DeclareOption{openacc}{\openacctrue}
\DeclareOption{onecolumn}{\onecolumntrue}
\DeclareOption{preview}{\previewtrue}
\DeclareOption{revised}{\revisedtrue}
\DeclareOption{draft}{\PassOptionsToPackage{draft}{graphicx}}
\DeclareOption{a3paper}{\PassOptionsToPackage{a3}{crop}}
\DeclareOption{centre}{\PassOptionsToPackage{center}{crop}}
\DeclareOption{crop}{\PassOptionsToPackage{cam}{crop}\global\@cropmarksontrue}
\DeclareOption{nocrop}{\PassOptionsToPackage{off}{crop}\global\@cropmarksonfalse}
\DeclareOption{info}{\PassOptionsToPackage{info}{crop}}
\DeclareOption{noinfo}{\PassOptionsToPackage{noinfo}{crop}}
\DeclareOption{numbered}{\global\@numberedreftrue}
\DeclareOption{named}{\global\@numberedreffalse}
\DeclareOption{noproofbox}{\global\@endproofboxfalse}
\DeclareOption{proofbox}{\global\@endproofboxtrue}
\DeclareOption{courier}{\def\verbatim@font{\normalfont\ttfamily}}
\DeclareOption{times}{\def\verbatim@font{\normalfont\rmfamily}}
\DeclareOption{endnotes}{\ProvidesPackage{endnotes}\note@endtrue}

\DeclareOption{vruler}{\global\@Vrulertrue}

\ExecuteOptions{info,proofbox,twocolumn,twoside}

\ProcessOptions

%\LoadClass[twoside,twocolumn]{article}

\RequirePackage[T1]{fontenc}
\RequirePackage{soul}
\RequirePackage{crop,graphicx,array,amsmath,amsthm,amssymb,amsfonts,endnotes}
\RequirePackage{upref}
\RequirePackage{upgreek}
\RequirePackage{color}
\RequirePackage{stfloats}

\if@Vruler
\usepackage{vruler-cob}
\linenos
\else
\RequirePackage{TI-lineno-COB}
\fi%

\usepackage{times}

\usepackage[figuresright]{rotating}


% Not sure if needed.
\newcommand\@ptsize{0}

% Set twoside printing
\@twosidetrue

% Marginal notes are on the outside edge
\@mparswitchtrue

% Default font sizes

\ifxipt
\renewcommand\normalsize{%
  \@setfontsize\normalsize{9.5\p@}{11\p@}%
  \abovedisplayskip 10\p@ \@plus 3\p@ \@minus4\p@
  \abovedisplayshortskip 3\p@ \@plus 3\p@ \@minus1\p@
  \belowdisplayshortskip 6\p@ \@plus 3\p@ \@minus2\p@
  \belowdisplayskip \abovedisplayskip
  \let\@listi\@listI}
\else
\renewcommand\normalsize{%
  \@setfontsize\normalsize{9.5\p@}{11\p@}%
  \abovedisplayskip 10\p@ \@plus 3\p@ \@minus4\p@
  \abovedisplayshortskip 3\p@ \@plus 3\p@ \@minus1\p@
  \belowdisplayshortskip 6\p@ \@plus 3\p@ \@minus2\p@
  \belowdisplayskip \abovedisplayskip
  \let\@listi\@listI}
\fi
\normalsize
\let\@bls\baselineskip

\newcommand\small{%
  \@setfontsize\small{9\p@}{11\p@}%
  \abovedisplayskip 6\p@ \@plus1\p@ \@minus1\p@
  \abovedisplayshortskip 3\p@ \@plus 2\p@ \@minus1\p@
  \belowdisplayshortskip 2\p@ \@plus 1\p@ \@minus1\p@
  \def\@listi{\leftmargin\leftmargini
              \listparindent \parindent
              \topsep 6\p@ \@plus2\p@ \@minus1\p@
              \parsep \z@ \itemsep \z@}%
  \belowdisplayskip \abovedisplayskip}

\newcommand\footnotesize{%
  \@setfontsize\footnotesize{8\p@}{9\p@}%
  \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
  \abovedisplayshortskip \z@ \@plus \p@
  \belowdisplayshortskip 1\p@ \@plus2\p@ \@minus2\p@
  \def\@listi{\leftmargin\leftmargini
              \listparindent 1em
              \topsep 5\p@ \@plus 2\p@ \@minus 1\p@
              \parsep \z@ \itemsep \z@}%
  \belowdisplayskip \abovedisplayskip}

\def\scriptsize{\@setfontsize\scriptsize{7\p@}{9\p@}}
\def\tiny{\@setfontsize\tiny{6\p@}{7\p@}}
\def\little{\@setfontsize\little{5\p@}{6\p@}}
\def\large{\@setfontsize\large{12\p@}{14\p@}}
\def\Large{\@setfontsize\Large{14\p@}{16\p@}}
\def\LARGE{\@setfontsize\LARGE{16\p@}{18\p@}}
\def\huge{\@setfontsize\huge{24\p@}{26\p@}}
\def\Huge{\@setfontsize\Huge{25\p@}{27\p@}}

\gdef\rec#1{\gdef\@rec{#1}}\rec{}
\gdef\rev#1{\gdef\@rev{#1}}\rev{}
\gdef\acc#1{\gdef\@acc{#1}}\acc{}


\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}


% Line spacing
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}

% Paragraph dimensions and inter-para spacing
\setlength\parskip{0\p@}
\setlength\parindent{4mm}

% Set inter-para skips
\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}

% Page break penalties
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301

% Disallow widows and orphans
\clubpenalty 10000
\widowpenalty 10000

% Disable page breaks before equations, allow pagebreaks after
% equations and discourage widow lines before equations.
\displaywidowpenalty 100
\predisplaypenalty   10000
\postdisplaypenalty  2500

% Allow breaking the page in the middle of a paragraph
\interlinepenalty 0

% Disallow breaking the page after a hyphenated line
\brokenpenalty 10000

% Hyphenation; don't split words into less than three characters
\lefthyphenmin=3
\righthyphenmin=3

% Set page layout dimensions
\setlength\headheight{6.3\p@}
\setlength\topmargin{3pc}
\addtolength\topmargin{0\p@}
\addtolength\topmargin{-1in}

\setlength\topskip{8\p@}
\setlength\headsep{30\p@}
\addtolength\headsep{-\topskip}

\setlength\footskip{26\p@}
\setlength\maxdepth{.5\topskip}

\setlength\textwidth{43pc}

\setlength\textheight{60\@bls}
\addtolength\textheight{\topskip}

\setlength\marginparsep{6\p@}
\setlength\marginparpush{6\p@}
\setlength\marginparwidth{10pt}

\setlength\oddsidemargin{4pc}%
\addtolength\oddsidemargin{-1in}
\setlength\@tempdima{\paperwidth}
\addtolength\@tempdima{-\textwidth}
\addtolength\@tempdima{-4pc}
\setlength\evensidemargin{\@tempdima}
\addtolength\evensidemargin{-1in}

\setlength\columnsep{5mm}
\setlength\columnseprule{0\p@}

% Footnotes
\setlength\footnotesep{0\p@}
\setlength{\skip\footins}{12\p@ \@plus 6\p@ \@minus 1\p@}

% Float placement parameters

\setcounter{totalnumber}{10}
% The maximum number of floats at the top and bottom of a page.
\setcounter{topnumber}{5}
\setcounter{bottomnumber}{5}

% The maximum part of the top or bottom of a text page that can be
% occupied by floats. This is set so that at least four lines of text
% fit on the page.
\renewcommand\topfraction{.921}
\renewcommand\bottomfraction{.921}

% The minimum amount of a text page that must be occupied by text.
% This should accomodate four lines of text.
\renewcommand\textfraction{.079}

% The minimum amount of a float page that must be occupied by floats.
\renewcommand\floatpagefraction{.887}

% The same parameters repeated for double column output
\renewcommand\dbltopfraction{.88}
\renewcommand\dblfloatpagefraction{.88}

% Space between floats
\setlength\floatsep    {18\p@ \@plus 2\p@ \@minus 2\p@}
% Space between floats and text
\setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
% Space above and below an inline figure
\setlength\intextsep   {18\p@ \@plus 2\p@ \@minus 2\p@}

% For double column floats
\setlength\dblfloatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}

% Space left at top, bottom and inbetween floats on a float page.
\setlength\@fptop{0\p@}         % no space above float page figures
\setlength\@fpsep{12\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}

% The same for double column
\setlength\@dblfptop{0\p@}
\setlength\@dblfpsep{12\p@ \@plus 1fil}
\setlength\@dblfpbot{0\p@ \@plus 2fil}

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Article Identification %%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\@default@journal@title{Journal Title}
\let\@journal@title\@default@journal@title
\def\jname@info{}
\def\jabbr@info{}
\def\art-id@info{}
\def\month@info{Month~}
\def\year@info{\the\year}
\def\vol@info{00}
\def\price@info{00}
\def\issue@info{00}
\def\@print@issn{0000-0000}
\def\@price{0000-0000}

\newcommand{\page@extent@info}{%
  \thepage--\pageref{LastPage}}

\newcommand{\totalpage}{%
  \pageref{LastPage}}

\newcommand{\issn@line}{\@print@issn}
\newcommand{\copyright@line}{\textcopyright\ \year@info\. Published by The Company of Biologists Ltd}
\gdef\doi@string{0000000000}

% Article information user commands
\def\jname#1{\gdef\jname@info{#1}}
\def\jabbr#1{\gdef\jabbr@info{#1}}
\def\Year#1{\gdef\year@info{#1}}
\def\Vol#1{\gdef\vol@info{#1}}
\def\Price#1{\gdef\price@info{#1}}
\def\Issue#1{\gdef\issue@info{#1}}
\def\art-id#1{\gdef\art-id@info{#1}}
\def\Month#1{\gdef\month@info{#1}}
\newcommand\DOI[2][\@doi@fixed@field]{%
  \gdef\@doi@variable@field{#1}%
  \setDOI{#1}{#2}}
\newcommand{\setDOI}[2]{%
  \gdef\doi@string{#1#2}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Firstpage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\firstpage}[1]{%
  \setcounter{page}{#1}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%% Page Styles %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\ps@titlepage{%
\vspace*{-10pt}
 \let\@evenfoot\relax
 \def\@evenhead{\fontsize{7}{9}\selectfont \titlepageheadline}
 \def\@oddhead{\hbox{\fontsize{7}{9}\selectfont \titlepageheadline}}%
 \def\@oddfoot{\hfill{\fontsize{7}{9}\selectfont\thepage}}%\numfont
 \def\authormark##1{\def\leftmark{##1}}%
 \def\titlemark##1{\markboth{##1}{##1}}}

\def\ps@headings{%
 \def\@oddfoot{\hfill{\fontsize{7}{9}\selectfont\thepage}}%\numfont
 \def\@evenfoot{\hfill{\fontsize{7}{9}\selectfont\thepage}}%\numfont
  \def\@evenhead{\let\etal=\tetal\fontsize{7}{9}\selectfont{\vbox{\hbox to \textwidth{{\color{titcolor}{\fontfamily{\sfdefault}\fontshape{sc}\selectfont
  \lowercase\@supertitle}}\hspace*{\fill}\jourinf}\vskip-2pt\hbox{\noindent\rule{\textwidth}{.5pt}}}}}
  \def\@oddhead{\let\etal=\tetal\fontsize{7}{9}\selectfont{\vbox{\hbox to \textwidth{{\color{titcolor}{\fontfamily{\sfdefault}\fontshape{sc}\selectfont
  \lowercase\@supertitle}}\hspace*{\fill}\jourinf}\vskip-2pt\hbox{\noindent\rule{\textwidth}{.5pt}}}}}
  \def\authormark##1{\def\leftmark{##1}}%
  \def\titlemark##1{\markboth{##1}{##1}}}

\def\ps@myheadings{%
  \def\@oddfoot{\hfill{\fontsize{9}{10}\selectfont\thepage}}%
  \let\@evenfoot\@empty%
  \def\@evenhead{\hfil}%
  \def\@oddhead{\let\etal=\tetal\fontsize{9}{10}\selectfont\hfil{}\hfil\llap{\phantom{09}}}%
  \def\authormark##1{\def\leftmark{##1}}%
  \def\titlemark##1{\markboth{##1}{##1}}}

\def\slinf{\hbox to \textwidth{Nova Techset Composition Pvt. Ltd.\hfill\artname \hfill Page\#:\quad \totalpage\hfill  Printed:\ \today}}

\def\cobcopyright{$\copyright$ \year@info. MANUSCRIPT SUBMITTED TO \MakeUppercase{\jname@info}\ (\year@info) \vol@info, \jabbr@info\art-id@info. doi:10.1242/\jabbr@info.\art-id@info}

\newcommand{\titlepageheadline}{%
  \vbox to 2\baselineskip{%
    \hbox to \textwidth{%
      \parbox{\textwidth}{%                   <- textwidth minus logo width
  \fontfamily{\sfdefault}\fontsize{7}{11}\selectfont
        {\cobcopyright\hfill \raisebox{-5pt}{\includegraphics{COB_Publisher_Logo}}\\[-3.5pt]
        \noindent \rule{\textwidth}{.5pt}%
}}}}}

\def\today{\number\day/\ifcase\month\or
 1\or 2\or 3\or 4\or 5\or 6\or 7\or 8\or 9\or 10\or 11\or 12\fi/\number\year}

%
% Sectional units
%

% Counters
\newcounter{part}
\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]
\newcounter{asection}[section]
\newcounter{refsection}[section]
\newcounter{appsection}[section]
\newcounter{appsubsection}[section]
\newcounter{figure}
\newcounter{scheme}
\newcounter{table}

% Form of the numbers
\newcommand\thepage{\arabic{page}}
\renewcommand\thesection{\Roman{section}}
\renewcommand\thesubsection{{\Alph{subsection}}}
\renewcommand\thesubsubsection{{\arabic{subsubsection}}}
\renewcommand\theparagraph{\thesubsubsection.\arabic{paragraph}}
\renewcommand\thesubparagraph{\theparagraph.\arabic{subparagraph}}
\renewcommand\theequation{\arabic{equation}}

% Form of the words
\newcommand\contentsname{Contents}
\newcommand\guestname{GUEST EDITORS}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\partname{Part}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{ABSTRACT}
\newcommand\refname{References}
\newcommand\bibname{References}
\newcommand\normsname{Norms}
\newcommand\indexname{Index}
\newcommand\figurename{Fig.}
\newcommand\schemename{Scheme}
\newcommand\tablename{Table}

% Clearemptydoublepage should really clear the running heads too
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{empty}\cleardoublepage}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Titling Macros %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Reset default command signatures

\def\title{\@ifnextchar[{\@title@wst@}{\@title@nst@}}
\def\@title@wst@[#1]{\titlemark{\lowercase{{#1}}}\@title@assign@}
\def\@title@nst@#1{\titlemark{#1}\@title@assign@{#1}}
\def\@title@assign@#1{\def\@title{#1}}

\def\author{\@ifnextchar[{\@author@wst@}{\@author@nst@}}
\def\@author@wst@[#1]{\authormark{{{#1}}}\@author@assign@}
\def\@author@nst@#1{\authormark{{{#1}}}\@author@assign@{#1}}
\def\@author@assign@#1{\gdef\@author{{#1}}\global\@author@providedtrue}


\def\and{\MakeLowercase{and}\space\ignorespaces}

% Ifs for titling commands
\newif\if@supertitle@provided \@supertitle@providedfalse
\newif\if@specialissue@provided \@specialissue@providedfalse
\newif\if@guesteditor@provided \@guesteditor@providedfalse
\newif\if@subtitle@provided \@subtitle@providedfalse
\newif\if@author@provided \@author@providedfalse
\newif\if@date@provided \@date@providedfalse
\newif\if@dedi@provided \@dedi@providedfalse
\newif\if@classification@provided \@classification@providedfalse
%\newif\if@abstract@provided \@abstract@providedfalse %%5/7/2004
%\newif\if@keywords@provided \@keywords@providedfalse

\def\keywordsname{KEYWORDS}

\def\date#1{\gdef\@date{#1}\global\@date@providedtrue}
\def\dedi#1{\def\@dedi{#1}\global\@dedi@providedtrue}
\def\keywords#1{{\if@Vruler\else\addtocounter{linenumber}{-1}\fi%%
\vskip7pt\noindent\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\fontsize{8}{9}\selectfont\keywordsname: #1\par\vskip-5.5pt
\ifonecolumn\noindent\rule{\textwidth}{1pt}\else\noindent\rule{20.75pc}{1pt}\fi}}%\global\@keywords@providedtrue
\def\subtitle#1{\def\@subtitle{{#1}}\global\@subtitle@providedtrue}
\def\supertitle#1{\def\@supertitle{{\noindent #1}}\global\@supertitle@providedtrue}
\def\specialissue#1{\def\@specialissue{{\noindent SPECIAL ISSUE:~#1}}\global\@specialissue@providedtrue}
\def\guesteditor#1{\def\@guesteditor{{\noindent \guestname:~#1}}\global\@guesteditor@providedtrue}
\def\@abbreviations{}
\def\@classification{}
\newcommand\classification[2][Jel Classification]{%
     \global\@classification@providedtrue
     \gdef\@classification{{\leftskip=2pc\rightskip=2pc \noindent\textit{#1}:\enskip #2\par}}}

%\newbox\@abstractbox
\newenvironment{abstract}{%\global\@abstract@providedtrue%%% 5/7/2004
\if@Vruler\else\addtocounter{linenumber}{1}\fi  %\global\setbox\@abstractbox
  \vbox\bgroup\fontfamily{\sfdefault}\fontsize{9}{11}\fontshape{m}\selectfont\if@Vruler\else\internallinenumbers\fi
{\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\selectfont\abstractname}\par
\vskip-6pt
\ifonecolumn\noindent\rule{\textwidth}{.5pt}\else\noindent\rule{20.75pc}{.5pt}\fi\par
%  \leftskip8mm\rightskip8mm
  \@afterindentfalse\@afterheading}{\par\egroup}%

\def\@fnsymbol#1{\ifcase#1\or *\or \dagger\or \ddagger\or
    \S\or \P\or \|\or **\or \dagger\dagger
    \or \ddagger\ddagger \else\@ctrerr\fi}

\mathchardef\Register="20AE%
\mathchardef\ter="099%
\def\titdagger{\hbox{$\dagger$}}
\def\titddagger{\hbox{$\ddagger$}}
\def\titS{\hbox{$\S$}}
\def\titP{\hbox{$\P$}}

%% Reimplement Titling Layout
\newcommand\maketitle{\par
      \begingroup
\renewcommand{\@fnsymbol}[1]{\ifcase##1\or \hbox{*}\or \titdagger\or \titddagger\or
    \titS\or \titP\or \|\or \hbox{**}\or \titdagger\titdagger
    \or \titddagger\titddagger \else\@ctrerr\fi}
    \thispagestyle{titlepage} %%
%\linenumbers
    \setcounter{footnote}{0}
    \renewcommand\thefootnote{}%
    \def\@makefnmark{\hbox{\rm\@thefnmark}}%
    \long\def\@makefntext##1{%\noindent
      {${\@thefnmark}$}{##1}}%
    \global\@topnum\z@   % Prevents figures from going at top of page.
\ifonecolumn
    \@maketitle
    \else
\twocolumn[\@maketitle]
\fi
    \@thanks
  \endgroup
  \@afterindentfalse
  \@afterheading
    \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
    \global\let\@thanks\@empty%
\if@Vruler\else\let\c@linenumber\c@runninglinenumber\addtocounter{linenumber}{-1}\linenumbers\fi}

\def\@maketitle{\if@Vruler\else\internallinenumbers\fi%
  \ifx\@journal@title\@default@journal@title
  \ClassError{COB}{Journal Code not specified. Processing stopped.}
  {The journal code is a required field. Your LaTeX file must contain\MessageBreak
   \protect\journal{JJJJ}, where JJJJ is the four letter journal code.}%
  \fi
\if@supertitle@provided%
\ifonecolumn\vspace*{-.1pt}\else\vspace*{15pt}\fi
       {\color{titcolor}{\fontfamily{\sfdefault}\fontsize{10}{12}\fontseries{b}\fontshape{sc}\selectfont\lowercase\@supertitle\par%
\vspace*{2.5pt}}}
    \else
       \vskip10pt%
     \fi
%%%%%%%%%%
  \begingroup%
  \vspace*{5pt}
    {\fontfamily{\sfdefault}\noindent\fontsize{18}{22}\fontseries{m}\raggedright\selectfont%
      {\@title}%
      \par}%
     \if@subtitle@provided%
       \vspace{20\p@}%
       {\raggedright\fontsize{14}{16}\fontseries{b}\raggedright\selectfont\itshape\MakeUppercase\@subtitle\par%
        \vspace{23.5\p@}}%
     \else%
       \vspace{5\p@}%
     \fi%
    \begingroup%
    \if@author@provided%
        {{\fontfamily{\sfdefault}\noindent\fontsize{12}{14}\fontseries{b}\fontshape{n}\selectfont\mathversion{bold}\raggedright\@author}\par}%
   \else%
       \vspace{0\p@}%
   \fi%
    \endgroup%
\endgroup
    \ifx\@address\empty\else
     \thanks{{\noindent\ifonecolumn\protect\rule{\textwidth}{.5pt}\else\protect\rule{20.75pc}{.5pt}\fi\newline\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont\@address}}%
    \fi%
    \ifx\@corres\empty\else
        \thanks{{\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont\@corres}\vskip1pc}%
    \fi%
%%%%%%%%%%%
%%%%\if@abstract@provided%
%%%%\vspace*{20pt}
%%%%    {\fontsize{9}{11}\selectfont\noindent\unvbox\@abstractbox\par}%
%%%%    \else
%%%%       \vspace{0\p@}%
%%%%     \fi
%%%%\if@keywords@provided%
%%%% \vspace*{20pt}
%%%%      {\parindent=0pt\fontsize{9}{11}\selectfont
%%%%      \textbf{\keywordsname:}\enskip \@keywords\par}%
%%%%    \vspace{4\p@}%
%%%%\else%
%%%%       \vspace{0\p@}%
%%%%\fi%
 \if@date@provided%
   \vspace*{7.5pt}%
    {{\thanks{\noindent\fontsize{8}{10}\fontseries{b}\fontshape{n}\selectfont \ifonecolumn\protect\rule{\textwidth}{.5pt}\else\protect\rule{20.75pc}{.5pt}\fi\newline\@date\ifx\@rec\empty\else Received \@rec;\ \fi \ifx\@rev\empty\else Revised \@rev\fi}}\par}%
    \vspace{4.5\p@}%%
   \else
    \vspace{0\p@}%
  \fi
     \if@dedi@provided%
       {\fontsize{8}{10}\fontseries{rm}\selectfont{\@dedi}\par
        \vspace{18.25\p@}}%
     \else
       \vspace{0\p@}%
     \fi
    \if@classification@provided
      {\leftskip\z@\parfillskip=\z@ plus1fill%
        \fontsize{8}{9}\selectfont\@classification\par}%
    \fi
  \par
  \vspace{1pc plus 4\p@ minus 2\p@}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Lastpage %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\usepackage{lastpage}
%% Last page number - from lastpage.sty
%\def\lastpage@putlabel{\addtocounter{page}{-1}%
%   \immediate\write\@auxout{\string
%   \newlabel{LastPage}{{}{\thepage}}}%
%   \addtocounter{page}{1}}
%\AtEndDocument{%
%   \message{Setting LastPage}%
%   \clearpage\lastpage@putlabel}%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\sodef\strack{}{0.185em}{0.75em}{0pt}
\sodef\runtrack{}{0.05em}{0.35em}{0pt}
\sodef\thmtrack{}{0.09em}{\z@}{\z@}


%%%%%%%%%%%%%%%%%%%%%%%%%%% Sectioning macros %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Lowest level heading that takes a number by default
\setcounter{secnumdepth}{0}

% Reset the default section macros to help with title casing
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
    \let\@svsec\@empty
  \else
    \refstepcounter{#1}%
    \protected@edef\@svsec{\@seccntformat{{#1}}\relax}%
  \fi
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@
    \begingroup
      #6{%
        \noindent{\hskip #3\relax\@svsec}%
          \interlinepenalty \@M
% Modified to uppercase H1's and titlecase the rest
          \ifnum #2>1\relax
            {#8}%
          \else
            \MakeUppercase{#8}%
          \fi
        \@@par}%
    \endgroup
    \csname #1mark\endcsname{#7}%
    \addcontentsline{toc}{#1}{%
      \ifnum #2>\c@secnumdepth \else
        \protect\numberline{\csname the#1\endcsname}%
      \fi
      #7}%
  \else
    \def\@svsechd{%
      #6{\hskip #3\relax
      \@svsec #8.\enspace }%
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}}%
  \fi
  \@xsect{#5}}

\newcommand{\sectionfont}{\fontfamily{\sfdefault}\fontsize{8.5}{11}\selectfont\bfseries\raggedright}
% H1

\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-12\p@ plus -3\p@ minus -3\p@}%
                                   {0.0001\p@}%
                                   {\sectionfont}}
% H2
\def\subsection{%
\renewcommand{\@seccntformat}[1]{{{\csname the##1\endcsname)\enskip}}}
  \@startsection{subsection}{2}{\z@}
  {-10\p@ plus -1\p@ minus -2\p@}{5.5\p@}
  {\fontfamily{\sfdefault}\fontsize{8.5}{11}\selectfont\bfseries\raggedright}}


% H3
\def\subsubsection{%
\renewcommand{\@seccntformat}[1]{{{\csname the##1\endcsname)\enskip}}}
  \@startsection{subsubsection}{3}{\z@}
  {-9.5\p@ plus -1\p@ minus -2\p@}{0.5\p@}
  {\fontfamily{\sfdefault}\fontsize{8.5}{11}\selectfont\raggedright}}

\def\paragraph{%
  \@startsection{paragraph}{4}{\z@}
  {-11\p@ plus -1\p@}{0.5\p@}
  {\fontsize{9.5}{11}\selectfont\itshape}}


 % H5
 \def\subparagraph{%
   \@startsection{subparagraph}{5}{\z@}
   {-11\p@ plus -1\p@}{0\p@}
   {\fontsize{9.5}{11}\selectfont\itshape}}

\def\refsection{%
  \@startsection{section}{1}{\z@}
  {-12\p@ plus -3\p@ minus -3\p@}{0.0001\p@}
  {\fontfamily{\sfdefault}\fontseries{b}\fontsize{8}{10}\selectfont}}

\def\appsection{%
  \@startsection{section}{1}{\z@}
  {-24\p@ plus -2\p@ minus -1\p@}{0.0001pt}
  {\fontsize{11.5}{13.5}\selectfont\bfseries}}

\def\appsections{%
  \@startsection{section}{1}{\z@}
  {-24\p@ plus -2\p@ minus -1\p@}{14pt}
  {\fontsize{11.5}{13.5}\selectfont\bfseries}}

\def\appsubsection{%
  \@startsection{section}{1}{\z@}
  {-12\p@ plus -2\p@ minus -1\p@}{0.0001\p@}
  {\fontsize{11.5}{13.5}\selectfont\itshape}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Footnotes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand\footnoterule{\begingroup\leftskip\z@
\noindent\vspace{0\p@}\endgroup}

\def\footnotealign{\hskip12pt}

\newcommand\@makefntext[1]{%
\fontsize{8}{9}\selectfont
    \parindent 1em%
    \noindent
    %\@hangfrom
    {\footnotealign\llap{\@makefnmark}}#1}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% From ifmtarg.sty
% Copyright Peter Wilson and Donald Arseneau, 2000
\begingroup
\catcode`\Q=3
\long\gdef\@ifmtarg#1{\@xifmtarg#1QQ\@secondoftwo\@firstoftwo\@nil}
\long\gdef\@xifmtarg#1#2Q#3#4#5\@nil{#4}
\long\gdef\@ifnotmtarg#1{\@xifmtarg#1QQ\@firstofone\@gobble\@nil}
\endgroup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%% PostScript Rotation Macros %%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@rotate \@rotatefalse
\newif\if@rotatecenter \@rotatecenterfalse
\def\rotatecenter{\global\@rotatecentertrue}
\def\rotateendcenter{\global\@rotatecenterfalse}
\def\rotate{\global\@rotatetrue}
\def\endrotate{\global\@rotatefalse}
\newdimen\rotdimen
\def\rotstart#1{\special{ps: gsave currentpoint currentpoint translate
    #1 neg exch neg exch translate}}
\def\rotfinish{\special{ps: currentpoint grestore moveto}}
\def\rotl#1{\rotdimen=\ht#1\advance\rotdimen by \dp#1
    \hbox to \rotdimen{\vbox to\wd#1{\vskip \wd#1
    \rotstart{270 rotate}\box #1\vss}\hss}\rotfinish}
\def\rotr#1{\rotdimen=\ht #1\advance\rotdimen by \dp#1
    \hbox to \rotdimen{\vbox to \wd#1{\vskip \wd#1
    \rotstart{90 rotate}\box #1\vss}\hss}\rotfinish}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% Figures and Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newlength{\abovecaptionskip}
\newlength{\belowcaptionskip}
\setlength{\abovecaptionskip}{6\p@}
\setlength{\belowcaptionskip}{2\p@}

\newbox\figurecaptionbox
\long\def\@makecaption#1#2{\vspace{\abovecaptionskip}\par%
  \setbox\figurecaptionbox=\hbox{\fontfamily{\sfdefault}\fontsize{7.5}{9}\selectfont
    {\bfseries#1}\enskip{#2}}%
  \ifdim\wd\figurecaptionbox<\hsize
\centerline{\box\figurecaptionbox}
  \else
    \fontfamily{\sfdefault}\fontsize{7.5}{9}\selectfont
    \unhbox\figurecaptionbox
    \par
  \fi}

\newbox\schemecaptionbox
\long\def\@schemecaption#1#2{\vspace{\abovecaptionskip}\par%
  \setbox\schemecaptionbox=\hbox{#1\enskip #2}%
  \ifdim\wd\schemecaptionbox<\hsize
{\box\schemecaptionbox}
  \else
\fontsize{8}{9}\selectfont
    \unhbox\schemecaptionbox
    \par
  \fi}

\newbox\tablecaptionbox
\long\def\@tablecaption#1#2{\par%
\setbox\tablecaptionbox=\hbox{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont
 {\bfseries #1}\enskip {#2}} %
  \ifdim\wd\tablecaptionbox<\hsize%
\centerline{\box\tablecaptionbox}
\vskip3pt
  \else
    \fontfamily{\sfdefault}\fontsize{8}{10}\selectfont
    \centering{\unhbox\tablecaptionbox}
\vskip3pt
    \par
  \fi}

\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{{\figurename} \thefigure.}
\def\figure{\@float{figure}}
\def\endfigure{\end@float}
\@namedef{figure*}{\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}

\def\thescheme{\@arabic\c@scheme}
\def\fps@scheme{h}
\def\ftype@scheme{2}
\def\ext@scheme{los}
\def\fnum@scheme{\MakeUppercase{\schemename}~\thescheme.}
\def\scheme{\let\@makecaption\@schemecaption\@float{scheme}}
\def\endscheme{\end@float}
\@namedef{scheme*}{\let\@makecaption\@schemecaption\@dblfloat{scheme}}
\@namedef{endscheme*}{\end@dblfloat}

\def\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{3}
\def\ext@table{lot}
\def\fnum@table{{\tablename}\enspace\thetable.}
\def\table{\let\@makecaption\@tablecaption\@float{table}}
\def\endtable{\end@float}
\@namedef{table*}{\let\@makecaption\@tablecaption\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}

\newdimen\tempdime
\newbox\temptbox

\newlength{\abovefigskip}
\newlength{\belowfigskip}
\setlength{\abovefigskip}{\z@}
\setlength{\belowfigskip}{\z@}

\def\toprule{\noalign{\ifnum0=`}\fi
  \hrule \@height 1\p@
  \hrule \@height 4\p@ \@width \z@
  \futurelet \@tempa\@xhline}

\def\midrule{\noalign{\ifnum0=`}\fi
  \hrule \@height 4.5\p@ \@width \z@
  \hrule \@height 0.5\p@
  \hrule \@height 5\p@ \@width\z@
  \futurelet \@tempa\@xhline}

\def\botrule{\noalign{\ifnum0=`}\fi
  \hrule \@height 4\p@ \@width \z@
  \hrule \@height 1\p@
  \futurelet \@tempa\@xhline}

% Table and array parameters
\setlength\arraycolsep{.5em}
\setlength\tabcolsep{.5em}
\setlength\arrayrulewidth{.25pt}
\setlength\doublerulesep{2.5pt}
\renewcommand\arraystretch{1}

\def\hrulefill{\leavevmode\leaders\hrule height .25pt\hfill\kern\z@}

\newcolumntype{P}[1]{>{\raggedright\let\\\@arraycr\hangindent1em}p{#1}}

\newenvironment{processtable}[3]{\setbox\temptbox=\hbox{{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont#2}}%
\tempdime\wd\temptbox\@processtable{#1}{#2}{#3}{\tempdime}}%
{\relax}

\newenvironment{@processtable}[4]{%
\if!#1!%
\addtocounter{table}{-1}%%ADDED ON 21/12/05
\def\fnum@table{}%
\else%
\def\fnum@table{{\tablename}\enspace\thetable.}%
\fi%
\if@rotate
\setbox4=\vbox to \textwidth{\vss\hbox to \textheight{%
\begin{minipage}{#4}%
\caption{#1}{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont#2}%
\@ifmtarg{#3}{}{\vskip7pt\noindent
\parbox{#4}{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont #3\par}}%
\end{minipage}}\vss}
\else
\setbox4=\hbox to \hsize{{\hss\begin{minipage}[t]{#4}%
\caption{#1}{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont#2}
\@ifmtarg{#3}{}{\vskip7pt\noindent
\parbox{#4}{\fontfamily{\sfdefault}\fontsize{8}{10}\selectfont #3\par}}
\end{minipage}\hss}}
\fi
\if@rotate\rotr{4}\else\box4\fi}
{\relax}

\newcommand{\tabhead}[1]{\textbf{#1}}%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Lists %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\labelenumi{({\rm\roman{enumi}})}
\def\theenumi{\arabic{enumi}}
\def\labelenumii{{\rm(\alph{enumii})}}
\def\theenumii{\alph{enumii}}
\def\p@enumii{\theenumi}
\def\labelenumiii{{\rm(\arabic{enumiii})}}
\def\theenumiii{\roman{enumiii}}
\def\p@enumiii{\theenumi(\theenumii)}
\def\labelenumiv{{\rm(\arabic{enumiv})}}
\def\theenumiv{\Alph{enumiv}}
\def\p@enumiv{\p@enumiii\theenumiii}
\def\labelitemi{$\bullet$}
\def\labelitemii{$\bullet$}
\def\labelitemiii{$\bullet$}
\def\labelitemiv{$\bullet$}

\setlength{\leftmargini}{11pt}
\setlength{\leftmarginii}{15pt}
\setlength{\leftmarginiii}{\z@}
\setlength{\leftmarginiv}{\z@}

\def\enumargs{%
   \partopsep     \z@
   \itemsep       0\p@
   \parsep        \z@
   \leftmargin    0pt
   \rightmargin   \z@
   \labelsep      -0.85em
   \listparindent \z@
   \itemindent    \z@}

\def\enumerate{%
    \@ifnextchar[{\@numerate}{\@numerate[(3)]}}

\def\@numerate[#1]{%
     \ifnum \@enumdepth >3 \@toodeep\else
     \advance\@enumdepth \@ne
     \edef\@enumctr{enum\romannumeral\the\@enumdepth}
     \list{\csname label\@enumctr\endcsname}{%
       \enumargs
       \setlength{\leftmargin}{\csname leftmargin\romannumeral\the\@listdepth\endcsname}
       \usecounter{\@enumctr}
       \settowidth\labelwidth{#1}
       \addtolength{\leftmargin}{\labelwidth}
       \addtolength{\leftmargin}{\labelsep}
       \def\makelabel##1{\llap{##1}}}%
     \fi
   }
\def\endenumerate{\endlist\advance \@listdepth -1}


\def\romenumargs{%
   \partopsep     \z@
   \itemsep       0\p@
   \parsep        \z@
   \leftmargin    15.2pt
   \rightmargin   \z@
   \labelsep      -1.2em
   \listparindent \z@
   \itemindent    \z@}

\def\romenumerate{%
    \@ifnextchar[{\@numerate}{\@romnumerate[(iii)]}}

\def\@romnumerate[#1]{%
     \ifnum \@enumdepth >3 \@toodeep\else
     \advance\@enumdepth \@ne
     \edef\@enumctr{enum\romannumeral\the\@enumdepth}
     \list{\csname label\@enumctr\endcsname}{%
       \romenumargs
       \setlength{\leftmarginii}{\csname leftmargin\romannumeral\the\@listdepth\endcsname}
       \usecounter{\@enumctr}
       \settowidth\labelwidth{#1}
       \addtolength{\leftmargin}{\labelwidth}
       \addtolength{\leftmargin}{\labelsep}
       \def\makelabel##1{\llap{##1}}}%
     \fi
   }
\def\endromenumerate{\endlist\advance \@listdepth -1}


% Changes to the list parameters for itemize
\def\itemargs{%
   \partopsep     \z@
   \itemsep       0pt
   \parsep        \z@
   \rightmargin   \z@
   \labelsep      0.5em
   \listparindent \parindent
   \itemindent    \z@}

\def\itemize{%
    \@ifnextchar[{\@itemize}{\@itemize[$\bullet$]}}

\def\@itemize[#1]{%
     \ifnum \@itemdepth >3 \@toodeep\else
     \advance\@itemdepth \@ne
     \edef\@itemctr{item\romannumeral\the\@itemdepth}
     \list{\csname label\@itemctr\endcsname}{%
       \itemargs
       \setlength{\leftmargin}{7.1pt}
       \def\makelabel##1{\hss \llap{##1}}}%
     \fi
   }
\let\enditemize\endlist

\setlength{\topsep}{6\p@}% \@plus 1\p@ \@minus 1\p@}
\setlength{\itemsep}{\z@}

\def\@listI{\leftmargin\leftmargini\topsep6.5\p@ plus 4\p@}
\let\@listi\@listI
\@listi
\def\@listii{\leftmargin\leftmarginii\topsep\z@}
\def\@listiii{\leftmargin\leftmarginiii\topsep\z@}
\def\@listiv{\leftmargin\leftmarginiv\topsep\z@}
\def\@listv{\leftmargin\leftmarginv\topsep\z@}
\def\@listvi{\leftmargin\leftmarginvi\topsep\z@}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% unnumbered list %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{description}{%
  \begin{list}{}{%
      \usecounter{enumi}%
      \setlength{\itemsep}{\z@}%
      \setlength{\labelwidth}{\z@}%
      \setlength{\leftmargin}{1em}%
      \setlength{\itemindent}{-1em}}}%
{\end{list}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Extracts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\quotation{\par\begin{list}{}{%\fontsize{9}{10}\selectfont%  Extracts in small type
      \setlength{\topsep}{\medskipamount}%
      \setlength{\itemsep}{\z@}%
      \setlength\labelwidth{\z@}%
      \setlength\labelsep{\z@}%
      \setlength\leftmargin{18pt}%
      \setlength\rightmargin{18pt}%
      \listparindent \parindent}%
  \item[\vrule width\z@ height 9\p@]}
\def\endquotation{\end{list}}
\let\quote\quotation
\let\endquote\endquotation
\newcommand{\source}[1]{\par{\textit{Source}: {\rm #1}}}
\newcommand{\note}[1]{\par{\textit{Note}: {\rm #1}}}
\newcommand{\notess}[1]{\par{\textit{Notes}: {\rm #1}}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% programs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

{\obeyspaces\global\let =\ }

\newskip\cmskipamount%
\cmskipamount=12pt plus 6pt minus 3pt%

\def\cmskip{\removelastskip\vskip\cmskipamount}

\long\def\bCM{\begingroup\parindent=0pt\cmskip\ttfamily%
\obeylines\obeyspaces}%

\def\eCM{\cmskip\endgroup}%

\long\def\tbCM{\begingroup\parindent=0pt\cmskip%
\obeylines\obeyspaces}%

\def\teCM{\cmskip\endgroup}%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\proofname}{\textit{Proof}}

\renewcommand{\qedsymbol}{\ensuremath{\square}}
\renewenvironment{proof}[1][\proofname]{\par
  \if@endproofbox\pushQED{\qed}\fi%
  \normalfont
  \topsep\bigskipamount
  \labelsep=0.5em\relax
  \trivlist
  \item[\quad\hskip\labelsep
        {#1}{\rm:}]\ignorespaces
}{%
  \if@endproofbox\popQED\fi
  \endtrivlist\@endpefalse
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Counter for saving the section numbering depth
\newcounter{savesecnumdepth}

%%%%%%%%%%%%%%%%%%%%%%%%%% Acknowledgments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\providecommand{\ackname}{\strack{ACKNOWLEDGEMENTS}}
\newcommand{\ack}[1][\ackname]{\normalsize\par%
  \setcounter{savesecnumdepth}{\value{secnumdepth}}%
  \setcounter{secnumdepth}{0}%
  \section*{#1}%
  \setcounter{secnumdepth}{\value{savesecnumdepth}}%
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\ifnote@end\let\footnote\endnote\fi%

\renewcommand{\notesname}{\strack{Notes}}

\newenvironment{notes}[1][Notes]{\par
\setcounter{secnumdepth}{0}
\section{\strack{#1}}%
}{\par}

%%%%%%%%%%%%%%%%%%%%%%%%% Nomenclature %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\providecommand{\nomenname}{Nomenclature}
\newcommand{\nomen}[1][\nomenname]{%
  \setcounter{savesecnumdepth}{\value{secnumdepth}}%
  \setcounter{secnumdepth}{0}%
  \refsection{#1}%
\vspace*{5pt}
  \setcounter{secnumdepth}{\value{savesecnumdepth}}%
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newenvironment{thebibliography}[1]{%
  \begingroup
  \fontsize{7}{9}\selectfont
  \setcounter{secnumdepth}{0}
  \refsection*{\bibname}%
  \begin{list}{[\arabic{enumi}]}{%
      \usecounter{enumi}%
      \setlength{\itemsep}{5pt}
      \setlength\leftmargin{\z@}%
      \setlength{\labelsep}{0.5em}%%chgd. 1em to 0.5em on 21/7/2004 -- devi
      \settowidth{\labelwidth}{[#1]}%
      \addtolength{\leftmargin}{\labelwidth}%
      \addtolength{\leftmargin}{\labelsep}}}%
{\end{list}\endgroup}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%\usepackage{hyperref}
%%%\hypersetup{
%%%    colorlinks,%
%%%    citecolor=blue,%
%%%    filecolor=blue,%
%%%    linkcolor=blue,%
%%%    urlcolor=blue
%%%}

\ifnumbib
\usepackage[sort&compress]{natbib}
\bibpunct{[}{]}{,}{n}{,}{;}

\renewenvironment{thebibliography}[1]{%
  \begingroup
  \fontsize{7}{9}\selectfont
  \setcounter{secnumdepth}{0}
  \refsection*{\bibname}%
  \begin{list}{[\arabic{enumi}]}{%
      \usecounter{enumi}%
      \setlength{\itemsep}{5pt}
      \setlength\leftmargin{\z@}%
      \setlength{\labelsep}{0.5em}%%chgd. 1em to 0.5em on 21/7/2004 -- devi
      \settowidth{\labelwidth}{[#1]}%
      \addtolength{\leftmargin}{\labelwidth}%
      \addtolength{\leftmargin}{\labelsep}}}%
{\end{list}\endgroup}
\else
\usepackage[authoryear]{natbib}
\bibpunct[, ]{(}{)}{;}{a}{,}{;}
\renewenvironment{thebibliography}[1]{%
  \begingroup
  \fontsize{8}{10}\selectfont
  \setcounter{savesecnumdepth}{\value{secnumdepth}}%
  \setcounter{secnumdepth}{0}
  \refsection{\bibname}%
  \begin{list}{}{%
      \usecounter{enumi}%
      \setlength{\labelwidth}{0pt}%
      \setlength{\leftmargin}{8pt}%
      \setlength{\itemindent}{-8pt}
      }}%
{\end{list}\setcounter{secnumdepth}{\value{savesecnumdepth}}%
\endgroup}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newskip\normskip
\setlength\normskip{6pt}

\newenvironment{normes}{%
  \begingroup
  \footnotesize\parindent=0pt
  \setcounter{secnumdepth}{0}
  \vskip\normskip
  {\fontsize{10}{12}\selectfont\bf\normsname}%
  \begin{list}{}{%
      \usecounter{enumi}%
      \setlength{\itemsep}{\z@}%
      \setlength{\labelwidth}{0pt}%
      \setlength{\leftmargin}{2em}%
      \setlength{\itemindent}{-2em}}}%
{\end{list}\endgroup}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Index %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@restonecol
\def\theindex{\@mainmatterfalse
  \@restonecoltrue\if@twocolumn\@restonecolfalse\fi
  \begingroup\parindent \z@
  \addcontentsline{toc}{section}{Index}
  \columnseprule \z@
  \columnsep 1pc\section*{\indexname} %Subject or Author here
  \parskip\z@ plus .3\p@\relax\let\item\idxitem\fontsize{8\p@}{10\p@}\selectfont}
\def\idxitem{\par\raggedright\hangindent 3em}
\def\subitem{\par\penalty1000\raggedright\hangindent 3em\hspace*{1em}}
\def\subsubitem{\par\penalty1000\raggedright\hangindent 3em\hspace*{2em}}
\def\endtheindex{\if@restonecol\endgroup\onecolumn\else\clearpage\fi}
\def\indexspace{\par \vskip 18\p@ plus 12\p@ minus 12\p@\relax}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% Appendix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\ifappend % Are we in the Appendix?
\def\appendix{\fontsize{11.5}{13.5}\selectfont\par
  \setcounter{secnumdepth}{0}
  \renewcommand{\thesection}{\Alph{section}}
  \renewcommand{\@seccntformat}[1]{{\csname
  the##1\endcsname}\quad}
\def\thetable{\thesection\@arabic\c@table}
\def\thefigure{\thesection\@arabic\c@figure}
\renewcommand\theequation{\thesection\arabic{equation}}
        \setcounter{section}{0}
        \setcounter{table}{0}
        \setcounter{equation}{0}
        \setcounter{subsection}{0}
        \appendtrue}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Override settings in MathTime back to TeX defaults
\DeclareMathSizes{5}     {5}   {5}  {5}
\DeclareMathSizes{6}     {6}   {5}  {5}
\DeclareMathSizes{7}     {7}   {5}  {5}
\DeclareMathSizes{8}     {8}   {6}  {5}
\DeclareMathSizes{9}     {9}   {7}  {5}
\DeclareMathSizes{10}   {10}   {7}  {5}
\DeclareMathSizes{12}   {12}   {9}  {7}

%Math parameters
\setlength{\jot}{5\p@}
\skip\@mpfootins = \skip\footins
\fboxsep=6\p@
\fboxrule=.25\p@

\medmuskip=4mu% plus 2mu minus 4mu
\thickmuskip=5mu% plus 5mu

\vbadness=9999
\tolerance=9999
\doublehyphendemerits=10000
\doublehyphendemerits 640000
\finalhyphendemerits  1000000

\flushbottom
\frenchspacing
%\onecolumn
\pagestyle{headings}

% Author queries
\fboxsep=4\p@
\fboxrule=0.5\p@

\newcommand{\query}[2][0pt]{%
  \marginpar{%\vspace*{#1}%
    {\parbox{\marginparwidth}{%
     \raggedright\small{\bf #2}}}}}

\newcommand{\leftquery}[2][0pt]{%
  \marginpar{%\vspace*{#1}%
    {\parbox{\marginparwidth}{%
     \raggedright\small{\hskip5pc\bf #2}}}}}


\newenvironment{queries}[1][9.99]{%
  \begingroup
  \thispagestyle{empty}
  \advance\@enumdepth \@ne
  \edef\@enumctr{enum\romannumeral\the\@enumdepth}
  \def\labelenumi{\arabic{enumi})}
  \def\labelenumii{\alph{enumii})}
  \begin{list}{\csname label\@enumctr\endcsname}{%\small%
      \setlength{\topsep}{12\p@}%
    \setlength{\itemsep}{6\p@}%
      \setlength{\leftmargin}{\z@}%
      \settowidth{\labelwidth}{\bfseries#1}%
      \setlength{\labelsep}{1.5em}%
      \addtolength{\leftmargin}{\labelwidth}%
      \addtolength{\leftmargin}{\labelsep}
      \def\makelabel##1{\hss\llap{{##1}}}
    }}
{\end{list}\endgroup}

% Short-form coding
\newcommand\tetal{{{\rm et~al.}}}
\newcommand\etal{\textit{et~al.}}
\newcommand\eg{\textit{e.g.}}
\newcommand\ie{\textit{i.e.}}
\let\endash\textendash

\def\dbond{\raise0.18ex\hbox{$=$}}
\def\bond{\raise0.18ex\hbox{$-$}}

\def\endash{--}
\def\emdash{\,--\,}
\def\sdot#1{{\buildrel{\scriptscriptstyle\bullet}\over{\hbox{$#1$}}}}
\def\ssdot#1{{\buildrel{\scriptscriptstyle\bullet\bullet}\over{\hbox{$#1$}}}}

\def\refmsg{\noindent\hangindent=1em}%

\def\z{\phantom}

%%%For New Table

\newcount\mscount
\def\multispan#1{\omit \mscount#1\relax
  \loop\ifnum\mscount>\@ne \sp@n\repeat}
\def\sp@n{\span\omit\advance\mscount\m@ne}

\hyphenation{Table}
\hyphenation{Figure}
\hyphenation{Section}
\hyphenation{equation}
\hyphenation{Theorem}
\hyphenation{Lemma}
\hyphenation{Algorithm}
\hyphenation{Definition}
\hyphenation{Corollary}
\hyphenation{Proof}
\hyphenation{Remark}

\def\bcontinued{({\it continued\/})}
\def\tcontinued{Continued}

\def\dd{{\rm d}}
\def\ee{{\rm e}}

\newdimen\p@renwd

\def\spbordermatrix#1{\begingroup \m@th
  \setbox\z@\vbox{\def\cr{\crcr\noalign{\kern2\p@\global\let\cr\endline}}%
    \ialign{$##$\hfil\kern2\p@\kern\p@renwd&\thinspace\hfil$##$\hfil
      &&\quad\hfil$##$\hfil\crcr
      \omit\strut\hfil\crcr\noalign{\kern-\baselineskip}%
      #1\crcr\omit\strut\cr}}%
  \setbox\tw@\vbox{\unvcopy\z@\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{\unhbox\@ne\unskip\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{$\kern\wd\@ne\kern-\p@renwd\left[\kern-\wd\@ne
    \global\setbox\@ne\vbox{\box\@ne\kern2\p@}%
    \vcenter{\kern-\ht\@ne\unvbox\z@\kern-\baselineskip}\,\hskip-\bmskip\right]$}%
  \null\;\vbox{\kern\ht\@ne\box\tw@}\endgroup}


\def\sqbordermatrix#1{\begingroup \m@th
  \setbox\z@\vbox{\def\cr{\crcr\noalign{\kern2\p@\global\let\cr\endline}}%
    \ialign{$##$\hfil\kern2\p@\kern\p@renwd&\thinspace\hfil$##$\hfil
      &&\quad\hfil$##$\hfil\crcr
      \omit\strut\hfil\crcr\noalign{\kern-\baselineskip}%
      #1\crcr\omit\strut\cr}}%
  \setbox\tw@\vbox{\unvcopy\z@\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{\unhbox\@ne\unskip\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{$\kern\wd\@ne\kern-\p@renwd\left[\kern-\wd\@ne
    \global\setbox\@ne\vbox{\box\@ne\kern2\p@}%
    \vcenter{\kern-\ht\@ne\unvbox\z@\kern-\baselineskip}\,\right]$}%
  \null\;\vbox{\kern\ht\@ne\box\tw@}\endgroup}

\def\llbordermatrix#1{\begingroup \m@th
  \setbox\z@\vbox{\def\cr{\crcr\noalign{\kern2\p@\global\let\cr\endline}}%
    \ialign{$##$\hfil\kern2\p@\kern\p@renwd&\thinspace\hfil$##$\hfil
      &&\quad\hfil$##$\hfil\crcr
      \omit\strut\hfil\crcr\noalign{\kern-\baselineskip}%
      #1\crcr\omit\strut\cr}}%
  \setbox\tw@\vbox{\unvcopy\z@\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{\unhbox\@ne\unskip\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{$\kern\wd\@ne\kern-\p@renwd\left[\kern-\wd\@ne
    \global\setbox\@ne\vbox{\box\@ne\kern2\p@}%
    \vcenter{\kern-\ht\@ne\unvbox\z@\kern-\baselineskip}\,\right.$}%
  \null\;\vbox{\kern\ht\@ne\box\tw@}\endgroup}

\def\lbordermatrix#1{\begingroup \m@th
  \setbox\z@\vbox{\def\cr{\crcr\noalign{\kern2\p@\global\let\cr\endline}}%
    \ialign{$##$\hfil\kern2\p@\kern\p@renwd&\thinspace\hfil$##$\hfil
      &&\quad\hfil$##$\hfil\crcr
      \omit\strut\hfil\crcr\noalign{\kern-\baselineskip}%
      #1\crcr\omit\strut\cr}}%
  \setbox\tw@\vbox{\unvcopy\z@\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{\unhbox\@ne\unskip\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{$\left[\kern\wd\@ne\kern-\p@renwd\kern-\wd\@ne
    \global\setbox\@ne\vbox{\box\@ne\kern2\p@}%
    \vcenter{\kern-\ht\@ne\unvbox\z@\kern-\baselineskip}\,\right.$}%
  \null\;\vbox{\kern\ht\@ne\box\tw@}\endgroup}

\newdimen\bmskip
\bmskip=0pt

\def\rbordermatrix#1{\begingroup \m@th
  \setbox\z@\vbox{\def\cr{\crcr\noalign{\kern2\p@\global\let\cr\endline}}%
    \ialign{\hfil$##$\hfil\kern6\p@\kern\p@renwd&\thinspace\hfil$##$\hfil
      &&\quad\hfil$##$\hfil\crcr
      \omit\strut\hfil\crcr\noalign{\kern-\baselineskip}%
      #1\crcr\omit\strut\cr}}%
  \setbox\tw@\vbox{\unvcopy\z@\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{\unhbox\@ne\unskip\global\setbox\@ne\lastbox}%
  \setbox\tw@\hbox{$\left.\kern\wd\@ne\kern-\p@renwd\kern-\wd\@ne
    \global\setbox\@ne\vbox{\box\@ne\kern2\p@}
    \vcenter{\kern-\ht\@ne\unvbox\z@\kern-\baselineskip}\hskip-\bmskip\right]$}%
  \null\;\vbox{\kern\ht\@ne\box\tw@}\endgroup}

\chardef\permil="89%

\usepackage{textcomp}
\let\euro\texteuro

\newcommand{\address}[1]{\gdef\@address{#1}}\gdef\@address{}%
\newcommand{\corres}[1]{\gdef\@corres{{Authors for correspondence:} #1}}\gdef\@corres{}%

\def\name#1{#1\newline}
\def\affil#1{{\textsuperscript{#1}}}

\def\add#1#2{{\textsuperscript{#1}}{#2\newline}}
\def\ladd#1#2{{\textsuperscript{#1}}{#2}\vspace*{1.5pt}}
\def\fadd#1#2{{\textsuperscript{#1}}{#2}}
\def\email#1{#1}

\newcommand\emailrule[1]{Email: #1\par\noindent\protect\rule{82.5mm}{.5pt}}

\usepackage[splitrule]{footmisc}

\definecolor{mag}{cmyk}{0,.36,0,0}
\definecolor{titcolor}{cmyk}{0.75,0,0.05,0}

\usepackage{marginnote}

\definecolor{mag}{cmyk}{0,.36,0,0}
\def\surname#1{\fboxsep=1pt\colorbox{mag}{#1}}%

\def\runetal{\hbox{\fontsize{6.5}{9}\selectfont\it E\kern0.35pt T\  A\kern0.35pt L.}}

\newcommand{\aubio}[1]{\vspace{1\baselineskip}{\noindent #1}}

\def\@journal@title{{\it COB} (\year@info), page~\thepage\ of \pageref{LastPage}\ $\copyright$\ The Authors, \year@info}

\def\@print@issn{0269-9648/11}
\def\@doi@fixed@field{10.1242/\jabbr@info.}
\def\@doi@variable@field{00000000000000000000}
\def\jourinf{\jname@info\ (\year@info) \vol@info, \jabbr@info\art-id@info. doi:10.1242/\jabbr@info.\art-id@info}

\renewcommand{\copyright@line}{\textcopyright\ The Authors \year@info}

\def\ack#1{{\vskip7.5pt\noindent{\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\fontsize{7.5}{9}\selectfont Acknowledgements}\par\noindent\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont#1}}
\def\competing#1{{\vskip7.5pt\noindent{\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\fontsize{7.5}{9}\selectfont Competing interests}\par\noindent\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont#1}}
\def\contribution#1{{\vskip7.5pt\noindent{\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\fontsize{7.5}{9}\selectfont Contribution}\par\noindent\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont#1}}
\def\funding#1{{\vskip7.5pt\noindent{\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\fontsize{7.5}{9}\selectfont Funding}\par\noindent\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont#1}}
\def\data#1{{\vskip7.5pt\noindent{\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\fontsize{7.5}{9}\selectfont Data availability}\par\noindent\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont#1}}
\def\supplementary#1{{\vskip7.5pt\noindent{\fontfamily{\sfdefault}\fontseries{b}\fontshape{n}\fontsize{7.5}{9}\selectfont Supplementary}\par\noindent\fontfamily{\sfdefault}\fontsize{7}{9}\selectfont#1}}

\usepackage{url}
\urlstyle{rm}
\def\TCH#1{\textbf{#1}}
\art-id{xxxxxx}
\Year{2019}

\usepackage{subfig}

\makeatletter
\newbox\figurecaptionbox
\long\def\@makecaption#1#2{\vspace{\abovecaptionskip}\par%
  \setbox\figurecaptionbox=\hbox{\fontfamily{\sfdefault}\fontsize{7.5}{9}\selectfont
    {\bfseries#1}\enskip{#2}}%
  \ifdim\wd\figurecaptionbox<\hsize
\centerline{\fontfamily{\sfdefault}\box\figurecaptionbox}
  \else
    \fontsize{7.5}{9}\selectfont
    \unhbox\figurecaptionbox
    \par
  \fi}
\makeatother

\ifDEV
\jname{Development}
\jabbr{dev}
\else
\ifJEB
\jname{Journal of Experimental Biology}
\jabbr{jeb}
\else
\ifJCS
\jname{Journal of Cell Science}
\jabbr{jcs}
\else
\ifDMM
\jname{Disease Models \& Mechanisms}
\jabbr{dmm}
\else
\ifBIO
\jname{Biology Open}
\jabbr{bio}
\else
\ClassError{PLEASE INSERT THE RESPECTIVE JOURNAL OPTION}
\fi
\fi
\fi
\fi
\fi

\ifDEV
\def\JNAME{Development}
\def\JABBR{DEV}
\else
\ifJEB
\def\JNAME{Journal of Experimental Biology}
\def\JABBR{JEB}
\else
\ifJCS
\def\JNAME{Journal of Cell Science}
\def\JABBR{JCS}
\else
\ifDMM
\def\JNAME{Disease Models \& Mechanisms}
\def\JABBR{DMM}
\else
\ifBIO
\def\JNAME{Biology Open}
\def\JABBR{BIO}
\fi
\fi
\fi
\fi
\fi


\endinput
