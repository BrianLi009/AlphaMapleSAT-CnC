// program for generating dimacs file, given a dimacs file without the first line
// result is printed to stdout
using namespace std;
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>


int main(int argc, char const **argv)
{
    std::ifstream input = ifstream(argv[1]);
    int maxVar = -1;
    int nClauses = 0;
    // get maxVar and nClausess
    string line;
    while (getline(input, line))
    {
        if (strncmp(line.c_str(), "c\t", 2) == 0)
            continue;
        istringstream iss(line);
        string lit;
        while (std::getline(iss, lit, ' '))
        {
            int l = stoi(lit);
            if (l == 0)
             nClauses++;
            else
                maxVar = max(maxVar, abs(l));
        }
    }
    input.close();

    printf("p cnf %d %d\n", maxVar, nClauses);
    input = ifstream(argv[1]);
    std::cout << input.rdbuf();
    input.close();
    return 0;
}