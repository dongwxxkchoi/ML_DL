N = int(input())
X = int(input())
num_list = map(int, input().split())

for num in num_list:
    if num > X:
        print(num, end=' ')