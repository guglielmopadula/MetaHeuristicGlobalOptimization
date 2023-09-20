import numpy as np
from collections import deque
from copy import deepcopy
"This contains the codes of Essentials of Metauristic which can be used in real global optimization in single core (Chapters 1,2,3)"

np.random.seed(0)
def rastirin(x):
    return -(20+np.sum(x**2-10*np.cos(2*np.pi*x)))

def rastirin_grad(x):
    return -((2*x+20*np.pi*np.sin(2*np.pi*x)))

def rastirin_hessian(x):
    return np.diag(-((2-40*np.pi*np.cos(2*np.pi*x))))

###Algorithm 1, gradient ascent

def alg_1(f,df,alpha,n,eps,max_times):
    xold=np.random.rand(n)
    for i in range(max_times):
        x=xold+alpha*df(xold)
        if np.linalg.norm(df(x))<eps:
            continue
        xold=x
    return x

x=alg_1(rastirin,rastirin_grad,0.1,2,0.01,10000)
print("Alg 1 result is", x,rastirin(x))


###Algorithm 2, netwon method

def alg_2(f,df,dff,alpha,n,eps,max_times):
    xold=np.random.rand(n)
    for i in range(max_times):
        x=xold+alpha*(dff(xold)@df(xold))
        if np.linalg.norm(df(x))<eps:
            continue
        xold=x
    return x

x=alg_2(rastirin,rastirin_grad,rastirin_hessian,0.001,2,0.01,10000)
print("Alg 2 result is", x,rastirin(x))

 

###Algorithm 3, gradient ascent with restarts

def alg_3(f,df,alpha,n,eps,max_times):
    xold=np.random.rand(n)
    xstar=xold
    for _ in range(max_times):
        for i in range(max_times):
            x=xold+alpha*df(xold)
            if np.linalg.norm(df(x))<eps:
                continue
            xold=x
        if f(x)>f(xstar):
            xstar=x
        x=np.random.rand(n)
    return xstar

x=alg_3(rastirin,rastirin_grad,0.1,2,0.01,100)
print("Alg 3 result is", x,rastirin(x))

###Algorithm 4, hill climbing

def alg_4(candidate,tweak,quality,max_times):
    s=candidate()
    for _ in range(max_times):
        r=tweak(s.copy())
        if quality(r)>quality(s):
            s=r
    return s

s=alg_4(lambda: alg_1(rastirin,rastirin_grad,0.1,2,0.01,10000), lambda x: x+0.01*np.random.rand(2), rastirin,10000)

print("Alg 4 result is", s,rastirin(s))


###Algorithm 5, steepest ascent

def alg_5(candidate,tweak,quality,n,max_times):
    s=candidate()
    for _ in range(max_times):
        r=tweak(s.copy())
        for i in range(n-1):
            w=tweak(s.copy())
            if quality(r)>quality(w):
                r=w
        if quality(r)>quality(s):
            s=r
    return s

s=alg_5(lambda: alg_1(rastirin,rastirin_grad,0.1,2,0.01,10000), lambda x: x+0.01*np.random.rand(2), rastirin,5,10000)

print("Alg 5 result is", s,rastirin(s))


###Algorithm 6, steepest ascent hill climbing with replacement

def alg_6(candidate,tweak,quality,n,max_times):
    s=candidate()
    best=s.copy()
    for _ in range(max_times):
        r=tweak(s.copy())
        for i in range(n-1):
            w=tweak(s.copy())
            if quality(r)>quality(w):
                r=w
        s=r
        if quality(s)>quality(best):
            best=s.copy()
    return best

s=alg_6(lambda: alg_1(rastirin,rastirin_grad,0.1,2,0.01,10000), lambda x: x+0.01*np.random.rand(2), rastirin,5,10000)

print("Alg 6 result is", s,rastirin(s))

###alg 7: uniform rng
print("Alg 7 is not used")

###alg 8 uniform convolution 
def alg_8(v,p,r,min,max,l):
    for i in range(l):
            if p>=np.random.rand():
                flag=True
                while flag:
                    n=-r+2*r*np.random.rand()
                    if v[i]+n>=min and v[i]+n<=max:
                        v[i]=v[i]+n
                        flag=False
    return v

print("Alg 8 is not an optimization algorithm")
        

###alg 9: random search

def alg_9(quality,max_times):
    best=np.random.randn(2)
    for _ in range(max_times):
        s=np.random.randn(2)
        if quality(s)>quality(best):
            best=s.copy()
    return best

s=alg_9(rastirin,10000)
print("Alg 9 result is", s,rastirin(s))


##Hill climbing with random restarts
def alg_10(quality,tweak,n,max_time):
    s=np.random.rand(n)
    best=s
    tot_time=0
    while tot_time<max_time:
        time=np.random.poisson()
        for i in range(time):
            r=tweak(s.copy())
            if quality(r)>quality(s):
                s=r.copy()
        tot_time=tot_time+time
        if quality(s)>quality(best):
            best=s
    return best
    
s=alg_10(rastirin,lambda x: alg_8(x,0.5,2,-5.12,5.12,2),2,10000)
print("Alg 10 result is", s,rastirin(s))

            
###alg 11 gaussian convolution 
def alg_11(v,p,sigma,min,max,l):
    for i in range(l):
            if p>=np.random.rand():
                flag=True
                while flag:
                    n=sigma*np.random.randn()
                    if v[i]+n>=min and v[i]+n<=max:
                        v[i]=v[i]+n
                        flag=False
    return v

print("Alg 11 is not an optimization algorithm")

print("Alg 12 is not implemented")


#Alg 13 simulated annealing
def alg_13(candidate,init_temp,tweak,scheduling,quality,max_time):
    s=candidate()
    best=s.copy()
    t=init_temp()
    for i in range(max_time):
        r=tweak(s.copy())
        if quality(r)>quality(s):
            s=r.copy()
        else:
            if np.random.rand()<np.exp(quality(r)-quality(s)):
                s=r.copy()
        t=scheduling(t)
        if quality(s)>quality(best):
            best=s.copy()
    return best

s=alg_13(lambda: alg_1(rastirin,rastirin_grad,0.1,2,0.01,100),
         lambda: 300,
         lambda x: alg_11(x,0.5,1,-5.12,5.12,2),
         lambda t: 0.8*t,
         rastirin,
         1000)

print("Alg 13 result is", s,rastirin(s))


#Alg 14: tabu search


def alg_14(candidate,quality,tweak,l,epsilon,n,maxtime):
    s=candidate()
    best=s 
    L=deque()
    L.append(s)
    for _ in range(maxtime):
        if len(L)>l:
            _=L.pop()
        r=tweak(s.copy())
        for i in range(n-1):
            w=tweak(s.copy())
            if np.min(np.linalg.norm(w-np.array(list(L)),axis=1))>epsilon and (quality(w)>quality(r) or np.min(np.linalg.norm(r-np.array(list(L)),axis=1))<epsilon):
                r=w.copy()
        if np.min(np.linalg.norm(r-np.array(list(L)),axis=1))>epsilon:
            s=r.copy()
            L.append(r)
        if quality(s)>quality(best):
            best=s.copy()
    return best

s=alg_14(lambda: alg_1(rastirin,rastirin_grad,0.1,2,0.01,100),
         rastirin,
         lambda x: alg_11(x,0.5,10,-5.12,5.12,2),
         30,
         0.01,
         10,
         1000)

print("Alg 14 result is", s,rastirin(s))

def tweak_with_L(s,L,tweak):
    flag=True
    s_bak=s.copy()
    L_indexes=[x[0] for x in L]
    s=tweak(s_bak.copy())
    s_new=s.copy()
    new_feat=[]
    for i in range(len(s)):
        if L_indexes.count(i)>0:
            s_new[i]=s_bak[i]
        else:
            new_feat.append(i)
    return s,new_feat

#Alg 14: tabu search


def alg_15(candidate,quality,tweakL,l,epsilon,n,maxtime):
    s=candidate()
    best=s 
    L=[]
    c=0
    for _ in range(maxtime):
        c=c+1   
        L_bak=deepcopy(L)
        for x in L_bak:
            if x[1]-c>l:
                L.remove(x)
        r,new_feat_r=tweakL(s.copy(),L)
        for i in range(n-1):
            w,new_feat_w=tweakL(s.copy(),L)
            if quality(w)>quality(r):
                r=w.copy()
                new_feat_r=deepcopy(new_feat_w)
        s=r.copy()
        for x in new_feat_r:
            L.append([x,c])
        if quality(s)>quality(best):
            best=s.copy()
    return best

#Alg 15 feature based tabu search


s=alg_15(lambda: alg_1(rastirin,rastirin_grad,0.1,2,0.01,100),
         rastirin,
         lambda x,L: tweak_with_L(x,L,lambda x: alg_11(x,0.5,10,-5.12,5.12,2)),
         30,
         0.01,
         10,
         1000)

print("Alg 15 result is", s,rastirin(s))

###Alg 16: ILS with random restarts

def alg_16(quality,tweak,newshomebase,perturb,n,max_time):
    s=np.random.rand(n)
    h=s.copy()
    best=s.copy()
    tot_time=0
    while tot_time<max_time:
        time=np.random.poisson()
        for i in range(time):
            r=tweak(s.copy())
            if quality(r)>quality(s):
                s=r.copy()
        tot_time=tot_time+time
        if quality(s)>quality(best):
            best=s.copy()
        h=newshomebase(h,s)
        s=perturb(h.copy())
        return best

s=alg_16(rastirin,
         lambda x: alg_8(x,0.5,2,-5.12,5.12,2),
         lambda x,y: y if rastirin(y)>rastirin(x) else x,
        lambda x: alg_8(x,0.5,5,-5.12,5.12,2),
         2,
         10000)

print("Alg 16 result is", s,rastirin(s))



##Alg 17: Abstract GEA
def alg_17(build,fitness,breed,join,maxtime):
    p=build()
    best=-1
    fit_vect=np.zeros(len(p))
    for _ in range(maxtime):
        for i in range(len(p)):
            fit_vect[i]=fitness(p[i])
        for i in range(len(p)):
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        p=join(p,breed(p))
    return best

def generate_weigth_matrix(n):
    A=np.random.rand(n,n)
    for i in range(n):
        A[i]=A[i]/np.sum(A[i])
    return A



s=alg_17(lambda: np.random.rand(100,2),rastirin,lambda x: generate_weigth_matrix(100)@x,lambda x,y: y, 1000)
    
print("Alg 17 result is", s,rastirin(s))

## Alg 18:lambda-mu evolution strategy

def alg_18(fitness,n,mutate,mu,l,maxtime):
    p=np.random.rand(l,n)
    best=-1
    for _ in range(maxtime):
        fit_vect=np.zeros(len(p))
        for i in range(len(p)):
            fit_vect[i]=fitness(p[i])
        for i in range(len(p)):
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        q=p[np.argsort(fit_vect)]
        child=[]
        for q_i in q:
            for _ in range(mu//l):
                child.append(mutate(q_i.copy()))
        p=np.array(child)
    return best

s=alg_18(rastirin,2, lambda x: alg_11(x,0.5,1,-5.12,5.12,2),10,10,100) 
print("Alg 18 result is", s,rastirin(s))

#Alg 19: The mu+lambda evolution strategy
def alg_19(fitness,n,mutate,mu,l,maxtime):
    p=np.random.rand(l,n)
    best=-1
    for _ in range(maxtime):
        fit_vect=np.zeros(len(p))
        for i in range(len(p)):
            fit_vect[i]=fitness(p[i])
        for i in range(len(p)):
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        q=p[np.argsort(fit_vect)]
        child=[]
        for q_i in q:
            for _ in range(mu//l):
                child.append(q_i)
                child.append(mutate(q_i.copy()))
        p=np.array(child)
    return best

s=alg_19(rastirin,2, lambda x: alg_11(x,0.5,1,-5.12,5.12,2),10,10,10) 
print("Alg 19 result is", s,rastirin(s))

#Alg 20: The genetic algorithm
def alg_20(popsize, fitness, crossover,select,n, mutate, maxtimes):
    p=np.random.rand(popsize,2)
    best=-1
    for _ in range(maxtimes):
        fit_vec=np.random.rand(popsize,n)
        for i in range(len(p)):
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        Q=[]
        for i in range(popsize//2):
            pa=select(p)
            pb=select(p)
            ca,cb=crossover(pa.copy(),pb.copy())
            Q.append(mutate(ca))
            Q.append(mutate(cb))
        P=np.array(Q)
    return best

#Alg 23: one point crossover
def alg_23(v,w):
    c=np.random.randint(0,len(v))
    v_bak=v[c:]
    w_bak=w[c:]
    w[c:]=v_bak.copy()
    v[c:]=w_bak.copy()
    return v,w



s=alg_20(100,rastirin,alg_23,lambda x: x[np.random.randint(len(x))],2,lambda x: alg_11(x,0.5,1,-5.12,5.12,2),100) 
print("Alg 20 result is", s,rastirin(s))
print("Algorithm 21 is not implemented ")
print("Algorithm 22 is not implemented ")
print("Algorithm 23 is an optimization ")

#Alg 24: two point crossover
def alg_24(v,w):
    b=np.random.randint(0,len(v))
    c=np.random.randint(0,len(v))
    min=np.min([b,c])
    max=np.max([b,c])
    b=min
    c=max
    v_bak=v[b:,:c]
    w_bak=w[b:,:c]
    w[b:,:c]=v_bak.copy()
    v[b:,:c]=w_bak.copy()
    return v,w

print("Algorithm 24 is not an optimization algorithm")

#Alg 25: random crossover

def alg_25(v,w):
    b=np.random.binomial(len(v),0.5)
    v_bak=v_bak
    w_bak=w_bak
    v=b*v+(1-b)*w
    w=b*w+(1-b)*v
    return v,w


print("Algorithm 25 is not an optimization algorithm")

#Alg 26: permutation mutation
def alg_26(v):
    b=np.random.permutation(len(v))
    return v[b]


print("Algorithm 26 is not an optimization algorithm")

print("Algorithm 27 is not implemented ")

#Alg 28: line mutation

def alg_28(v,w,p):
    alpha=-p+(1+2*p)*np.random.rand()
    beta=-p+(1+2*p)*np.random.rand()
    v_bak=v.copy()
    w_bak=w.copy()
    v=(1-alpha)*v_bak+alpha*w_bak
    w=(1-beta)*w_bak+beta*v_bak
    return v,w

#Alg 29: intermediate line mutation

def alg_29(v,w,p):
    alpha=-p+(1+2*p)*np.random.rand(len(v))
    beta=-p+(1+2*p)*np.random.rand(len(v))
    v_bak=v.copy()
    w_bak=w.copy()
    v=(1-alpha)*v_bak+alpha*w_bak
    w=(1-beta)*w_bak+beta*v_bak
    return v,w

print("Algorithm 28 is not an optimization algorithm")
print("Algorithm 29 is not an optimization algorithm")
print("Algorithm 30 is not an optimization algorithm")
print("Algorithm 31 is not an optimization algorithm")

#Alg 32: tournament selection
def alg_32(P,fitness,t):
    best=P[np.random.randint(0,len(P))].copy()
    for i in range(t):
        tmp=P[np.random.randint(0,len(P))]
        if fitness(tmp)>fitness(best):
            best=tmp.copy()
    return tmp

print("Algorithm 32 is not an optimization algorithm")

#Alg 33:The genetic algorithm with elitism
def alg_33(popsize, fitness, crossover,select,n,n_elite, mutate, maxtimes):
    p=np.random.rand(popsize,n)
    best=-1
    for _ in range(maxtimes):
        fitness_vec=np.random.rand(popsize,n)
        for i in range(len(p)):
            fitness_vec[i]=fitness(p[i])
        for i in range(len(p)):
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        tmp=p[np.argsort(fitness_vec)]
        Q=[tmp[i].tolist() for i in range(n_elite)]
        for i in range((popsize-n)//2):
            pa=select(p)
            pb=select(p)
            ca,cb=crossover(pa.copy(),pb.copy())
            Q.append(mutate(ca))
            Q.append(mutate(cb))
        P=np.array(Q)
    return best

s=alg_33(100,
         rastirin,
         alg_23,
         lambda P: alg_32(P,rastirin,8),
         2,
         8,
         lambda x: alg_11(x,0.5,1,-5.12,5.12,2),
         100) 

print("Alg 33 result is", s,rastirin(s))

#Alg 34:The steady state genetic algorithm
def alg_34(popsize, fitness, crossover,select,n,n_elite, mutate, maxtimes):
    p=np.random.rand(popsize,n)
    best=-1
    for _ in range(maxtimes):
        fitness_vec=np.random.rand(popsize,n)
        for i in range(len(p)):
            fitness_vec[i]=fitness(p[i])
        for i in range(len(p)):
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        tmp=p[np.argsort(fitness_vec)]
        Q=[tmp[i].tolist() for i in range(n_elite)]
        for i in range((popsize-n)//2):
            pa=select(p)
            pb=select(p)
            ca,cb=crossover(pa.copy(),pb.copy())
            ca=mutate(ca)
            cb=mutate(cb)
            if fitness(ca)>fitness(best):
                best=ca.copy()
            if fitness(cb)>fitness(best):
                best=cb.copy()
            pd=select(p)
            p=p.tolist()
            p.remove(pd.tolist())
            p=np.array(p)
            pe=select(p)
            p=p.tolist()
            p.remove(pe.tolist())
            p.append(ca.tolist())
            p.append(cb.tolist())
            p=np.array(p)
    return best
s=alg_34(100,
         rastirin,
         alg_23,
         lambda x: x[np.random.randint(len(x))],
         2,
         8,
         lambda x: alg_11(x,0.5,1,-5.12,5.12,2),
         100) 
print("Alg 34 result is", s,rastirin(s))

#Alg 35: Genetic Algorithm (Tree-Style Genetic Programming Pipeline)
def alg_35(popsize, fitness, crossover,select,n,r, maxtimes):
    p=np.random.rand(popsize,n)
    best=-1
    for _ in range(maxtimes):
        fitness_vec=np.random.rand(popsize,n)
        for i in range(len(p)):
            fitness_vec[i]=fitness(p[i])
        for i in range(len(p)):
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        tmp=p[np.argsort(fitness_vec)]
        Q=[]
        flag=True
        while flag:
            if r>=np.random.rand():
                pe=select(p)
                Q.append(pe.tolist())
            else:
                pa=select(p)
                pb=select(p)
                ca,cb=crossover(pa.copy(),pb.copy())
                Q.append(ca)
                if len(Q)<popsize:
                    Q.append(cb)
            if len(Q)==popsize:
                flag=False
        P=np.array(Q)
    return best
s=alg_35(100,
         rastirin,
         alg_23,
         lambda x: x[np.random.randint(len(x))],
         2,
         0.5,
         100) 
print("Alg 35 result is", s,rastirin(s))

#Alg 36: An Abstract Hybrid Evolutionary and Hill-Climbing Algorithm

def alg_36(t,build,fitness,join,breed,n,max_time):
    p=build()
    best=-1
    for _ in range(max_time):
        fit_vec=np.zeros(len(p))
        for i in range(len(p)):
            fit_vec[i]=fitness(p[i])
        for i in range(len(p)):
            for j in range(t):
                p[i]=alg_4(lambda: p[i], lambda x: x+0.01*np.random.rand(2), rastirin,10000)
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
            p=join(p,breed(p))
    return best


s=alg_36(1,
         lambda: np.random.rand(10,2),
         rastirin,
         lambda x,y: y,
         lambda x: generate_weigth_matrix(10)@x,
         2, 
         1)

print("Alg 36 result is", s,rastirin(s))


#Alg 37: A simplified Scatter Search with Path Relinking
print("Alg 37 is not implemented")


#Alg 38: Differential evolution
def alg_38(popsize,alpha,fitness,crossover,n,max_time):
    p=np.random.rand(popsize,n)
    q=-1
    best=-1
    for _ in range(max_time):
        fit_vec=np.zeros(len(p))
        for i in range(len(p)):
            fit_vec[i]=fitness(p[i])
        for i in range(len(p)):
            if q is not -1 and fitness(q[i])>fitness(p[i]):
                p[i]=q[i].copy()
            if best is -1 or fitness(p[i])>fitness(best):
                best=p[i].copy()
        q=p.copy()
        for i in range(len(q)):
            a=q[np.random.randint(len(q))]
            b=q[np.random.randint(len(q))]
            c=q[np.random.randint(len(q))]
            d=a+alpha*(b-c)
            p[i]=crossover(d,q[i].copy())[0]
    return best

s=alg_38(50,0.5,rastirin,alg_23,2,100)

print("Alg 38 result is", s,rastirin(s))

#Alg 39: wikipedia version
def alg_39(swarmsize,fitness,alpha,beta,gamma,n,maxtime):
    x=-1+2*np.random.rand(swarmsize,n)
    g=x[0]
    p=x.copy()
    for i in range(swarmsize,n):
        if fitness(p[i])>fitness(g):
            g=p[i].copy()
    v=-2+4*np.random.rand(swarmsize,n)
    for _ in range(maxtime):
        r=np.random.rand(swarmsize,n)
        v=alpha*(v)+beta*(p-x)+gamma*(g-x)
        x=x+v
        for i in range(len(v)):
            if fitness(x[i])>fitness(p[i]):
                p[i]=x[i].copy()
            if fitness(p[i])>fitness(g):
                g=p[i].copy()
    return g

s=alg_39(50,rastirin,0.5,0.5,0.5,2,100)
print("Alg 39 result is", s,rastirin(s))

print("Best algorithms are PSO and DE")