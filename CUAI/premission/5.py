# get input. Don't cast to int because we'll use string indexing
N = input()
# default value is True
res = True

for i in range(len(N)//2):
    # check if it is palindrome
    if N[i] != N[len(N) - (i+1)]:
        # if not, result is False
        res = False

# print the result
print(res)
