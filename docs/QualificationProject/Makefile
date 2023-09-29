### Makefile
### $Id: Makefile,v 1.11 2011/01/17 14:13:29 tiago Exp $
all: uninstall qualificacao.dvi qualificacao.ps qualificacao.pdf clean

uninstall:
	rm -f qualificacao.pdf qualificacao.ps qualificacao.dvi

qualificacao.dvi: qualificacao.tex
		latex  qualificacao
		bibtex qualificacao
		latex  qualificacao
		latex  qualificacao

qualificacao.ps: qualificacao.dvi
		dvips -Z -Pamz -Pcmz -Ppdf -G0 qualificacao.dvi -o qualificacao.ps

qualificacao.pdf: qualificacao.ps
		ps2pdf -dOptimize=true -dEmbedAllFonts=true -dPDFSETTINGS=/printer qualificacao.ps

clean:
	rm -f core *.core *.log *.aux *.toc *.lo[fpta] *.blg *.bbl \
	*.ind *.ilg *.idx *.glo *.gls *.out *~ *.backup \
	includes/*.aux includes/*~ includes/*.backup

### Makefile ends here.
