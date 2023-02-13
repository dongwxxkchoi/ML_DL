# Cosine function
def Cosine():
    # get three inputs
    a = int(input())
    b = int(input())
    c = int(input())

    # Triangular Determination Criteria
    if a+b>c and a+c>b and b+c>a:
        # this is the formula that get cosine(A)
        res = (pow(b, 2) + pow(c, 2) - pow(a, 2)) / (2 * b * c)
        print(res)
    else:
        print("NO")


if __name__ == "__main__":
    # call Cosine function
    Cosine()

