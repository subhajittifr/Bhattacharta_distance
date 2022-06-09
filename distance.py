import numpy as np
from numpy.linalg import inv, det
from scipy.integrate import simps
from scipy import stats


def RANGE(x1,x2):
    return np.array([np.min([np.min(x1),np.min(x2)]), np.max([np.max(x1),np.max(x2)])])


def nor(xy,x,y):
    sum1=0
    for i in range(len(xy)):
        for j in range(len(xy)):
            
            deltax=x[i+1]-x[i]
            deltay=y[i+1]-y[i]
            
            sum1+=xy[i][j]*deltax*deltay
    return sum1

def bhatt_dis_gen(X1,Y1,X2,Y2,nbin=None):
    if (nbin==None):
        nbin=100
    ret1 = stats.binned_statistic_2d(X1,Y1, None, 'count', bins=nbin,range=[RANGE(X1,X2),RANGE(Y1,Y2)])
    ret2 = stats.binned_statistic_2d(X2,Y2, None, 'count', bins=nbin,range=[RANGE(X1,X2),RANGE(Y1,Y2)])
    hist1=ret1.statistic/nor(ret1.statistic,ret1.x_edge,ret1.y_edge)
    hist2=ret2.statistic/nor(ret2.statistic,ret2.x_edge,ret2.y_edge)
    arrx,arry=[],[]
    for i in range(nbin):
        arrx.append((ret1.x_edge[i+1]+ret1.x_edge[i])/2)
        arry.append((ret1.y_edge[i+1]+ret1.y_edge[i])/2)
    bhatt_coeff=simps(simps(np.sqrt(hist1*hist2),arrx),arry)
    return -np.log(bhatt_coeff)  


# def bhatt_dis_gen(X1,Y1,X2,Y2,nbin=None):
#     if (nbin==None):
#         nbin=100
#     ret1 = stats.binned_statistic_2d(X1,Y1, None, 'count', bins=nbin,range=[RANGE(X1,X2),RANGE(Y1,Y2)])
#     ret2 = stats.binned_statistic_2d(X2,Y2, None, 'count', bins=nbin,range=[RANGE(X1,X2),RANGE(Y1,Y2)])
#     hist1=ret1.statistic/nor(ret1.statistic,ret1.x_edge,ret1.y_edge)
#     hist2=ret2.statistic/nor(ret2.statistic,ret2.x_edge,ret2.y_edge)
#     deltax=ret1.x_edge[1]-ret1.x_edge[0]
#     deltay=ret1.y_edge[1]-ret1.y_edge[0]
#     bhatt_coeff=np.sum(np.sqrt(hist1*hist2)*deltax*deltay)
#     return -np.log(bhatt_coeff)  


def maha_dis(X1,Y1,X2,Y2):
    Sig1=np.cov(np.array([X1,Y1]))
    Sig2=np.cov(np.array([X2,Y2]))
    Mu1=np.array([np.mean(X1),np.mean(Y1)])
    Mu2=np.array([np.mean(X2),np.mean(Y2)])
    Mu=Mu1-Mu2
    Sig=Sig1+Sig2
    w=np.sqrt(np.dot(Mu,np.dot(inv(Sig),Mu)))
    return w




###########################################################################################################################################
#                                       Analytical Formulae
###########################################################################################################################################


def bhatt_dis(mean1,mean2,cov1,cov2):
    mean=mean1-mean2
    cov=(cov1+cov2)/2
    icov=inv(cov)
    p1=np.dot(icov,mean)
    p2=np.dot(mean,p1)
    w=(1/8*p2+1/2*np.log(det(cov)/np.sqrt(det(cov1)*det(cov2))))
    return w
