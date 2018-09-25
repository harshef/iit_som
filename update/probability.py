def probability(t, k, CN, RN, G):
    u = []
    beta = []
    u_sum = 0
    a = 0
    b = 0
    for i in range(0, k):
        if t < G:
            for j in range(0, t+1):
                a = a + RN[j][i]
                b = b + CN[j][i]
            u.insert(i, a/(float(b)))
            u_sum = u_sum + u[i]
        else:
            for k in range((t-G+1), t+1):
                a = a + RN[k][i]
                b = b + CN[k][i]
            u.insert(i, a/(float(b)))
            u_sum = u_sum + u[i]
    for ii in range(0, k):
        x = u[ii]
        y = u_sum
        beta.insert(ii, x/float(y))
    return beta