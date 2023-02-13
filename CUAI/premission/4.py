import math

# get two input, point_a and point_p
# this function calculate the distance between two point a and p
# return the value of distance whose type is float
def get_point_distance(point_a: tuple, point_p: tuple) -> float:
    x1, y1 = point_a[0], point_a[1]
    x2, y2 = point_p[0], point_p[1]
    # formula to get the distance between two points
    dist = math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
    return dist

# get three input, point_a, point_p1, and point_p2
# this function calculate the distance between point_a and the straight line
# the straight line contains two points, p1 and p2
# return the value of distance whose type is float
def get_line_distance(point_a: tuple, point_p1: tuple, point_p2: tuple) -> float:
    x1, y1 = point_a[0], point_a[1]
    x2, y2, x3, y3 = point_p1[0], point_p1[1], point_p2[0], point_p2[1]
    # if x3 == x2, we can't use the formula, because x3-x2 should be divider
    # In this case, straight line is x = x3 (or x = x2)
    # so distance is |x3 - x1| (or |x2 - x1|)
    if x3 == x2:
        dist = abs(x3 - x1)
    # I'll let the straight line y = mx + c
    else:
        # m: the straight line's gradient
        m = abs((y3-y2)/(x3-x2))
        # y = mx + c => y2 = mx2 + c => c = y2 - mx2
        c = -m * x2 + y2
        # this is the formula to get the distance between the point and straight line
        dist = abs(m * x1 - 1 * y1 + c) / math.sqrt(pow(m, 2) + pow(-1, 2))
    return dist


# get three input, point_a, point_p1, point_p2
# this function check if the two point p1 and p2 is at the same side based on the straight line
# who's gradient is orthogonal with the straight line who contains p1 and p2
# and this line contains point_a
# this is important because, if p1 and p2 are at the same side,
# the shortest distance between a and the segment p1p2 (not the straight line)
# is not the distance between a and the straight line
def checker(point_a: tuple, point_p1: tuple, point_p2: tuple) -> bool:
    x1, y1 = point_a[0], point_a[1]
    x2, y2, x3, y3 = point_p1[0], point_p1[1], point_p2[0], point_p2[1]
    # m is the gradient of the straight line l who's gradient is orthogonal
    # with the straight line who contains p1,p2 and contains point_a

    # if x3 == x2, y = y1 might be l
    if x3 == x2:
        m = 0
    # if y3 == y2, x = x1 might be l
    # so l's gradient can be approximated as math.inf
    elif y3 == y2:
        m = math.inf
    # formula
    else:
        m = -1 / abs((y3 - y2) / (x3 - x2))

    c = -m * x1 + y1
    # this comes from
    # dist = abs(『m * x1 - 1 * y1 + c』) / math.sqrt(pow(m, 2) + pow(-1, 2))
    d1 = m * x2 - 1 * y2 + c
    d2 = m * x3 - 1 * y3 + c

    # if d1 and d2 have same sign, it means two points are at the same side
    # so this case return False
    if d1*d2>0:
        return False
    # two points are at the other side
    # so this case return True
    else:
        return True


if __name__ == "__main__":
    # res will contain the distances between A and other points,
    # and A and segments
    res = list()

    x_A, y_A = map(float, input().split())
    x_C1, y_C1 = map(float, input().split())
    x_C2, y_C2 = map(float, input().split())
    x_C3, y_C3 = map(float, input().split())
    x_C4, y_C4 = map(float, input().split())

    # save values as tuple (like point's coordinate values)
    A = (x_A, y_A)
    points = [(x_C1, y_C1), (x_C2, y_C2), (x_C3, y_C3), (x_C4, y_C4)]

    # append distances at res list
    for i in range(len(points)):
        res.append(get_point_distance(A, points[i]))
        # if i == 3, it will call points[4] which not exists
        # call checker to check
        if i != 3 and checker(A, points[i], points[i+1]):
            res.append(get_line_distance(A, points[i], points[i+1]))

    # print with formatting
    if min(res) % 1 == 0:
        print(int(min(res)))
    else:
        print('%.2f' % min(res))