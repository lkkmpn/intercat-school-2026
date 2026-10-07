# set variable "CC" to be the name of your compiler's executable
CC=gfortran 

model : rate13main.o dvode.o subs.o dcodes.o freeze.o dcrates.o dcanalyse.o
	$(CC) -o model rate13main.o dvode.o subs.o dcodes.o freeze.o dcrates.o dcanalyse.o
		
rate13main.o : rate13main.f
	$(CC) -c -O2 rate13main.f

dcodes.o : dcodes.f
	$(CC) -c -O2 dcodes.f

freeze.o : freeze.f
	$(CC) -c -O2 freeze.f

dvode.o : dvode.f
	$(CC) -c -O2 dvode.f

subs.o : subs.f
	$(CC) -c -O2 subs.f
	
dcrates.o : dcrates.f
	$(CC) -c -O2 dcrates.f
	
dcanalyse.o : dcanalyse.f
	$(CC) -c -O2 dcanalyse.f

dc.out: model
	./model

dc.dat: dc.out dc.pl
	./dc.pl

# dc.png: dc.dat dc.R
# 	R CMD BATCH dc.R

clean:
	rm -f model
	rm -f rate13main.o dvode.o subs.o dcodes.o freeze.o dcrates.o dcanalyse.o
	rm -f dc.dat dc.out rates.txt rate13steady.state dc.png
	rm -f *~
	
