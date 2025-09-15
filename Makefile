all: dist-solve.cpp
	g++ -O4 dist-solve.cpp -o dist-solve

clean:
	rm -f instances/ks_17.cnf.*
	rm -f instances/ks_19.cnf.*
