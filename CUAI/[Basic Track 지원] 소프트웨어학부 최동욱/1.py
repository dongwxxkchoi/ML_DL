# cast str to int
N = int(input())
X = int(input())

# split input strings, and cast strs to int
num_list = map(int, input().split())

# print num in num_list if num is bigger than X
for num in num_list:
    if num >= X:
        print(num, end=' ')


