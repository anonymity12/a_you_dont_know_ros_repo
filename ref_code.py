import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats


np.set_printoptions(threshold=3)
np.set_printoptions(suppress=True)
import cv2


def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))

def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.LINE_AA 
    color = (r, g, b)
    ctrx = center[0,0]
    ctry = center[0,1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)
    

def mouseCallback(event, x, y, flags,null):
    global center
    #tt the real path of our mouse pointer
    global trajectory
    global previous_x
    global previous_y
    global zs
    
    center=np.array([[x,y]])
    #tt record the path (x,y) in a vertical array
    trajectory=np.vstack((trajectory,np.array([x,y])))
    #noise=sensorSigma * np.random.randn(1,2) + sensorMu
    #tt which is always true once you moved the mouse
    if previous_x >0:
        heading=np.arctan2(np.array([y-previous_y]), np.array([previous_x-x ]))

        if heading>0:
            heading=-(heading-np.pi)
        else:
            heading=-(np.pi+heading)
        #tt 算范数，like length, 计算了本次 mouse 的移动 距离
        distance=np.linalg.norm(np.array([[previous_x,previous_y]])-np.array([[x,y]]) ,axis=1)
        
        std=np.array([2,4])
        #tt u name as GPS is better
        u=np.array([heading,distance])
        #tt we can predict where all the particle are after mouse move by `predict`
        predict(particles, u, std, dt=1.)
        #tt z is robot 和 6 个landmarks 的传感器 距离。np.random.randn(NL) will give u 6 个来自 标正 的数据
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
        #tt R maybe 方差
        update(particles, weights, z=zs, R=50, landmarks=landmarks)
        #tt 从weight 中抽取一些新的 ele
        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)

    previous_x=x
    previous_y=y
    


WIDTH=800
HEIGHT=600
WINDOW_NAME="Particle Filter"

#sensorMu=0
#sensorSigma=3

sensor_std_err=5


def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles



def predict(particles, u, std, dt=1.):
    N = len(particles)
    #tt u[1] == distance, std[1] == 4
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    #tt update the particle's new position. we predict
    #tt cause we know the angle and distance
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist
   
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        #tt 就是开平方距离
        distance=np.power((particles[:,0] - landmark[0])**2 +(particles[:,1] - landmark[1])**2,0.5)
        #tt we fix weights by norm。 通过 正态分布 进行 w 的修正
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

 
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)
    
def neff(weights):
    return 1. / np.sum(np.square(weights))
##tt 这就是他说的窗口，3rd way
def systematic_resample(weights):
    N = len(weights)
    #tt 1->50 每个 加上一个 正的随机 0-1 偏移，then normalize 到 0-1 区间
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    #tt i + j 最后== 50（N）
    while i < N and j<N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes
    
def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var
#tt u can change name to resample_weights_by_index
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    #tt normalize the weights again
    weights /= np.sum(weights)

    
x_range=np.array([0,800])
y_range=np.array([0,600])

#Number of partciles
N=400

landmarks=np.array([ [144,73], [410,13], [336,175], [718,159], [178,484], [665,464]  ])
NL = len(landmarks)
particles=create_uniform_particles(x_range, y_range, N)


weights = np.array([1.0]*N)


# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME,mouseCallback)

center=np.array([[-10,-10]])

trajectory=np.zeros(shape=(0,2))
robot_pos=np.zeros(shape=(0,2))
previous_x=-1
previous_y=-1
DELAY_MSEC=50

while(1):

    cv2.imshow(WINDOW_NAME,img)
    img = np.zeros((HEIGHT,WIDTH,3), np.uint8)
    drawLines(img, trajectory,   0,   255, 0)
    drawCross(img, center, r=255, g=0, b=0)
    
    #landmarks
    for landmark in landmarks:
        cv2.circle(img,tuple(landmark),10,(255,0,0),-1)
    #tt stage 1 done: how: by: use the mean of all particle's x and y
    sum_x = 0
    sum_y = 0
    #draw_particles:
    for particle in particles:
        cv2.circle(img,tuple((int(particle[0]),int(particle[1]))),1,(255,255,20),-1)
        sum_x += int(particle[0])
        sum_y += int(particle[1])
   
    want_x = sum_x // len(particles)
    want_y = sum_y // len(particles)
    cv2.circle(img, (want_x, want_y), 4, (255,255,255), -1)

    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break
    
    cv2.circle(img,(10,10),10,(255,0,0),-1)
    cv2.circle(img,(10,30),3,(255,255,255),-1)
    cv2.putText(img,"Landmarks",(30,20),1,1.0,(255,0,0))
    cv2.putText(img,"Particles",(30,40),1,1.0,(255,255,255))
    cv2.putText(img,"Robot Trajectory(Ground truth)",(30,60),1,1.0,(0,255,0))

    drawLines(img, np.array([[10,55],[25,55]]), 0, 255, 0)
    


cv2.destroyAllWindows()