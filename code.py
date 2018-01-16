__author__ = 'zw'
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import random
from numpy import linalg as LA
from sklearn.metrics import average_precision_score
hierarchy=sio.loadmat('data/corel5k_hierarchy_structure.mat')
hierarchy_edge=hierarchy['corel5k_hierarchy_structure']['edgeMatrix'][0][0]
label_train=sio.loadmat('data/train_annot.mat')['v']
label_train=label_train.astype(float)
label_test=sio.loadmat('data/test_annot.mat')['v']
label_test=label_test.astype(float)
label_ori=np.concatenate((label_train,label_test),axis=0)
hierarchy_matrix=np.zeros((np.shape(label_train)[1],np.shape(hierarchy_edge)[0]))
for i in range(np.shape(hierarchy_edge)[1]):
    child=hierarchy_edge[i][0]
    parent=hierarchy_edge[i][1]
    hierarchy_matrix[child][i]=-1
    hierarchy_matrix[parent][i]=1
DataTrain=sio.loadmat('data/fileTrain.mat')['v']
DataTrain=preprocessing.MinMaxScaler().fit_transform(DataTrain)
#DataTrain=preprocessing.scale(DataTrain)
DataTest=sio.loadmat('data/fileTest.mat')['v']
#DataTest=preprocessing.scale(DataTest)
DataTest=preprocessing.MinMaxScaler().fit_transform(DataTest)
Z_train=label_train.copy()
Z_test=np.zeros(label_test.shape)
Y_test=np.zeros(label_test.shape)
Y_test=Y_test.astype(float)
Y_test.fill(0.5)
count=0
for iter_i in range(900):
    i=random.randint(0,4499)
    num=random.randint(30,150)
    count=count+num
    for iter_j in range(num):
        j=random.randint(0,259)
        label_train[i][j]=0.5
        Z_train[i][j]=0
#print(count)
#print(np.where(label_ori[:,:]==0.5))
X=np.concatenate((DataTrain,DataTest),axis=0)
Z=np.concatenate((Z_train,Z_test),axis=0)
Y=np.concatenate((label_train,Y_test),axis=0)
#print(np.where(Y[:,:]==0.5))
def Compute_dist(X,near_x):
    index=np.zeros([np.shape(X)[0],1])
    for i in range(np.shape(X)[0]):
        diff=np.zeros([np.shape(X)[0],1])
        for j in range(np.shape(X)[0]):
            if i==j:
                continue
            diff[j]=np.linalg.norm(X[i]-X[j])
        sorted_diff=sorted(diff)
        index[i]=sorted_diff[near_x]
        #print(index[i])
    np.save("index.npy",index)
#Compute_dist(X,7)
def compute_Lx(X):
    Wx = np.zeros([np.shape(X)[0], np.shape(X)[0]], dtype="float")
    Dx = np.zeros([np.shape(X)[0], np.shape(X)[0]], dtype="float")
    I = np.identity(np.shape(X)[0], dtype="float")
    dists = np.load('index.npy')
    for i in range(np.shape(X)[0]):
        for j in range(np.shape(X)[0]):
            fz=sum((X[i]-X[j])*(X[i]-X[j]))
            kernel_i=dists[i]
            kernel_j = dists[j]
            Wx[i][j]=np.exp(-1*fz/(kernel_i*kernel_j))
        Dx[i][i]=sum(Wx[i])
        #print(Dx[i][i])
    Dx_tmp=np.diag(1/np.sqrt(np.diag(Dx)))
    tmp1=np.dot(Dx_tmp,Wx)
    tmp1=np.dot(tmp1,Dx_tmp)
    Lx=I-tmp1
    np.save('Lx.npy',Lx)
    #sio.savemat('Lx.mat', Lx)
    return Lx

def compute_Lc(Y):
    Wc=np.zeros([np.shape(Y)[1],np.shape(Y)[1]],dtype="float")
    Dc=np.zeros([np.shape(Y)[1],np.shape(Y)[1]],dtype="float")
    I=np.identity(np.shape(Y)[1],dtype="float")
    for i in range(np.shape(Wc)[1]):
        for j in range(np.shape(Wc)[1]):
            if i==j:
                Wc[i][j]=0
            else:
                yi=np.sqrt(np.dot(Y[i],Y[i]))
                yj = np.sqrt(np.dot(Y[j], Y[j]))
                cos_angle=float(np.dot(Y[i],Y[j])/(yi*yj))
                if cos_angle-1<1e-8:
                    fz=90
                else:
                    angle = np.arccos(cos_angle)
                    fz = angle * 360 / (2 *np.pi)
                fmi = np.sqrt(sum(np.multiply(Y[i], Y[i])))
                fmj = np.sqrt(sum(np.multiply(Y[j], Y[j])))
                Wc[i][j] = fz / (fmi * fmj)
                #print(Wc[i][j])
        Dc[i][i]=sum(Wc[i])
    Dc_tmp=np.diag(1/np.sqrt(np.diag(Dc)))
    tmp1=np.dot(Dc_tmp,Wc)
    tmp1=np.dot(tmp1,Dc_tmp)
    Lc=I-tmp1
    return Lc

#Lx=compute_Lx(X)
#Lc=compute_Lc(2*Y-1)
#with hierarchy
def MLML_MG(Y, Lx, Lc, Z0, Phi, Option, beta, gama):
    Y_bar = 2 * Y - 1
    A_bar = -2*Y_bar
    B_bar = beta * Lx
    C_bar = gama * Lc
    D = Phi
    if type(D)==int:
        Z_new=MLML_without(A_bar, B_bar, C_bar, Z0, Option)
    else:
        Z_new=MLML_with(A_bar, B_bar, C_bar, Phi, Z0, Option, beta, gama)
    return Z_new
#without hierarchy
def MLML_without(A, B, C, Z, Option):
    max_iter = Option['max_iter']
    gap_compute = Option['gap_compute']
    rate_step = Option['rate_step']
    alpha_rate=Option['alpha_rate']
    obj = np.zeros([max_iter+1,1])
    obj[0] = compute_obj(A,B,C,Z)
    alpha = np.zeros((max_iter,1))
    Z_old=Z
    for i in range(max_iter):
        Z_gradient=A+2*np.dot(Z_old,B)+2*np.dot(C,Z_old)
        if (i/gap_compute)==np.fix(i/gap_compute):
            alpha[i]=compute_alpha(A,B,C,Z_old,Z_gradient)/(alpha_rate*(i+1))
        else:
            alpha[i]=alpha[i-1]*rate_step
        Z_new=Z_old-alpha[i]*Z_gradient
        Z_new[np.where(Z_new[:,:]>1)]=1.0
        Z_new[np.where(Z_new[:, :] <0)] = 0.0
        obj[i+1]=compute_obj(A,B,C,Z_new)
        obj_diff=abs((obj[i+1]-obj[i])/obj[i])
        if obj_diff<1e-7:
            #print('小于')
            break
        else:
            Z_old=Z_new
    return Z_new

def compute_obj(A,B,C,Z):
    return np.trace(np.dot(Z,A.transpose()))+np.trace(np.dot(np.dot(Z,B),Z.transpose()))+np.trace(np.dot(C,np.dot(Z,Z.transpose())))
def compute_alpha(A,B,C,Z,Z_gradient):
    fz=0.5*np.trace(np.dot(Z_gradient,A.transpose()))+np.trace(np.dot(np.dot(Z,B),Z_gradient.transpose()))+np.trace(np.dot(Z_gradient.transpose(),np.dot(C,Z)))
    fm=np.trace(np.dot(np.dot(Z_gradient,B),Z_gradient.transpose()))+np.trace(np.dot(Z_gradient.transpose(),np.dot(C,Z_gradient)))
    return fz/fm

def MLML_with(A, B, C, Phi, Z, Option, beta, gama):
    rho=Option['rho']
    max_iter = Option['max_iter']
    Z_old=Z
    jiao_old=np.zeros((np.shape(Phi)[1],np.shape(Z)[1]))
    Q_old=np.dot(Phi.transpose(),Z_old)
    Q_old[np.where(Q_old[:,:]<0)]=0
    B_bar=B
    for i in range(max_iter):
        A_bar = A + np.dot(Phi, jiao_old - rho * Q_old)
        C_bar = C + 0.5 * rho * np.dot(Phi, Phi.transpose())
        Z_new=MLML_without(A_bar,B_bar,C_bar,Z_old,Option,beta,gama)
        Q_new=np.dot(Phi.transpose(),Z_new)+jiao_old/rho
        Q_new[np.where(Q_new[:,:]<0)]=0
        jiao_new=jiao_old+rho*(np.dot(Phi.transpose(),Z_new)-Q_new)
        dist=np.dot(Phi.transpose(),Z_new)-Q_new
        dist=LA.norm(dist,'fro')
        if(dist<1e-8):
            break
        jiao_old=jiao_new
        Q_old=Q_new
        if i%Option['rho_gap']==0:
            rho=min(1e10,rho*Option['rho_rate'])
    return Z_new
Option={}
Option['max_iter']=200
Option['gap_compute']=3
Option['rate_step']=0.5
Option['alpha_rate']=3

Lc=compute_Lc(Y)
#print(np.where(Lc[:,:]==np.nan))
#Lx=compute_Lx(X)
Lx=np.load('Lx.npy')
#print(Lx[0])
Lc=preprocessing.scale(Lc)
Lx=preprocessing.scale(Lx)
beta=0.1
gama=0.001
'''
betas = [0.1, 1, 5, 10, 50]
gamas = [0, 0.01, 0.1, 1, 10]
for i in range(5):
    beta = betas[i]
    gama = gamas[i]
    Z_new=MLML_MG(Y.transpose(),Lx,Lc,Z.transpose(),0,Option,beta,gama)
    Z_new[np.where(Z_new[:,:]>1)]=1.0
    Z_new[np.where(Z_new[:, :] <0)] = 0.0
    print(i)
    print(average_precision_score(label_ori,Z_new.transpose()))
    print(average_precision_score(label_ori[4500:],Z_new.transpose()[4500:]))
'''
Z_new=MLML_MG(Y.transpose(),Lx,Lc,Z.transpose(),0,Option,beta,gama)
Option['rho']=10
Option['gap_compute']=3
#Option['alpha_rate']=3
Option['rho_gap']=10
Option['rho_rate']=10

#Z_new=MLML_MG(Y.transpose(),Lx,Lc,Z.transpose(),hierarchy_matrix,Option,beta,gama)
Z_new[np.where(Z_new[:,:]>1)]=1.0
Z_new[np.where(Z_new[:, :] <0)] = 0.0
print(average_precision_score(label_ori,Z_new.transpose()))
print(average_precision_score(label_ori[4500:],Z_new.transpose()[4500:]))





