from numpy import sqrt

def cal_minkowski_score(tru, sol):
    count1 = 0;
    count2 = 0;
    count3 = 0
    for i in range(len(tru)):
        for j in range(i + 1, len(tru)):
            if tru[i] == tru[j] and sol[i] == sol[j]:
                count1 += 1
            if tru[i] == tru[j] and sol[i] != sol[j]:
                count2 += 1
            if tru[i] != tru[j] and sol[i] == sol[j]:
                count3 += 1
    numerator = count2 + count3
    denominator = count1 + count2
    mini = (float(numerator) / denominator)
    mink = sqrt(mini)
    return mink