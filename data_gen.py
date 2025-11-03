import numpy as np
import poly_trajectory_mosek as pt
import time
def generate_points(num,min_dist,bound):
    points=[]
    point=np.array([np.random.uniform(0,min_dist),np.random.uniform(0,min_dist)])
    points.append(point)
    while len(points)<num:
        point_front=points[-1]
        point=np.array([np.random.uniform(0,bound),np.random.uniform(0,bound)])
        
        if point[0]-point_front[0]<min_dist and point[1]-point_front[1]<min_dist:
            points.append(point)
    return points
def data_generate(name,idx):
    points=generate_points(num=5,min_dist=1.5,bound=3)#最大速度限制为0.4m/s
    points.insert(0,np.array([0,0]))#起点
    points=np.array(points)
    print(points)
    dim=2
    knots=np.array([0.0,3.0,6.0,9.0,12.0,15.0])
    order=8
    optimTarget = 'end-derivative' #'end-derivative' 'poly-coeff'
    maxConti = 4
    objWeights = np.array([0, 0, 1])
    pTraj = pt.PolyTrajGen(knots, order, optimTarget, dim, maxConti)
    #pin
    ts = knots.copy()
    Xdot = np.array([0, 0])
    Xddot = np.array([0, 0])
    for i in range(points.shape[0]):
        pin_ = {'t':ts[i], 'd':0, 'X':points[i]}
        pTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':1, 'X':Xdot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[0], 'd':2, 'X':Xddot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[-1], 'd':1, 'X':Xdot,}
    pTraj.addPin(pin_)
    pin_ = {'t':ts[-1], 'd':2, 'X':Xddot,}
    pTraj.addPin(pin_)
    # solve
    pTraj.setDerivativeObj(objWeights)
    print("solving")
    time_start = time.time()
    pTraj.solve()
    time_end = time.time()
    print(time_end - time_start)
    if pTraj.isSolved:
        #获取轨迹点
        sampleTs = np.linspace(knots[0], knots[-1], num=1500)
        samplePs = pTraj.eval(sampleTs,0)
        samplePs=np.array(samplePs).T
        print(samplePs.shape)
        print("finish {}".format(idx))
        np.save(name,samplePs)
        return True
    else:
        print("not solved")
        return False
        
if __name__=='__main__':
    data_dir='/root/quad_rl/mobile_data_poly/'
    i=0
    while i<5000:
        name=data_dir+'{}.npy'.format(i)
        if data_generate(name,i):
            i+=1