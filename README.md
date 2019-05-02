### Description: This is the programming project for EECS 839 mining special data
- Utilizing MLEM2 algorithm to find a set of possible rules and a set of certain rules for a dataset
- Using discretization to transform numerical values to unique strings

[project link](https://people.eecs.ku.edu/~jerzygb/proj-839-19.pdf)

### Test files
```
| test.txt
| austr-aca-35.txt
| echo-35-s.txt
| iris-35-h.txt
| iris-35-qm.txt
| m-global.txt
| final.txt
```

### How to run
```python
python MLEM2.py <input file name> <alpha> <output file name>
```
e.x  python MLEM2.py final.txt 0.5 out.txt


### Additional Statements
1. An additional file is included to test my algorithm. If you select final.txt, then you can see all the intermediate steps print out in the terminal. If you check the rules, they might not match exactly as the homework solution and that is because the order of attribute-value pairs are not organized the same way as they were on the homework. For example, if when two attribute-value pairs have a tie, then the first one will be picked thus this results in slightly different rule sets.
