import math

def Cosine():
    a = int(input())
    b = int(input())
    c = int(input())

    if a+b>c and a+c>b and b+c>a:
        res = (pow(b, 2) + pow(c, 2) - pow(a, 2)) / (2 * b * c)
        print(res)
    else:
        print("NO")

Cosine()