from utilities import load_points_from_file as lpfs, view_data_segments as vds
from utilities import *

def xgp(x, n):
    x1 = np.ones(x.shape)
    for i in range(1, (n + 1)):
        x1 = np.c_[x1, np.power(x, i)]
    return(x1)

def ygp(x1, a):
    y = a[0]
    for i in range(1, a.size):
        y = y + a[i]*x1[:,i]
    return(y)

def xgt(x):
    x1 = np.c_[np.ones(x.shape), np.sin(x)]
    return(x1)

def ygt(x1, a):
    y = a[0] + a[1]*x1[:,1]
    return(y)

def lsr(x1, y):
    xt = x1.transpose()
    a = np.linalg.inv(np.dot(xt, x1)).dot(xt).dot(y)
    return(a)

def sse(y, yh):
    e = np.power((yh - y), 2).sum()
    return(e)

args = sys.argv[1:]
xs, ys = lpfs(args[0])
te = 0
for i in range(0, int(xs.size/20)):
    x = xs[i*20:(i + 1)*20]
    y = ys[i*20:(i + 1)*20]
    x1l = xgp(x, 1)
    x1c = xgp(x, 3)
    x1t = xgt(x)
    al = lsr(x1l, y)
    ac = lsr(x1c, y)
    at = lsr(x1t, y)
    yhl = ygp(x1l, al)
    yhc = ygp(x1c, ac)
    yht = ygt(x1t, at)
    e = [sse(y, yhl), sse(y, yhc), sse(y, yht)]
    te = te + np.min(e)
    f = np.argmin(e)
    if (len(args) == 2):
        if (args[1] == "--plot"):
            xp = (np.linspace(x.min(), x.max(), 100))
            if (f == 0):
                x1p = xgp(xp, 1)
                yhp = ygp(x1p, al)
            if (f == 1):
                x1p = xgp(xp, 3)
                yhp = ygp(x1p, ac)
            if (f == 2):
                x1p = xgt(xp)
                yhp = ygt(x1p, at)
            plt.plot(xp, yhp)

if (len(args) == 2):
    if (args[1] == "--plot"):
        vds(xs, ys)
print(te)
