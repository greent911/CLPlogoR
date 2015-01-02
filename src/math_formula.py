from math import atan, degrees
def computeRelativeAngle(siftangle, vx, vy):
    angle = 0.0
    rangle = 0.0
    if vx > 0 and vy >= 0:
        angle = degrees(atan(vy/vx))        
    elif vx == 0 and vy == 0:
        angle = 0.0
    elif vx == 0 and vy > 0:
        angle = 90.0
    elif vx < 0 and vy > 0:
        angle = 90.0 + degrees(atan(-vx/vy))        
    elif vx < 0 and vy == 0:
        angle = 180.0       
    elif vx < 0 and vy < 0:
        angle = 180.0 + degrees(atan(vy/vx))        
    elif vx == 0 and vy < 0:
        angle = 270.0
    elif vx > 0 and vy < 0:
        angle = 270.0 + degrees(atan(vx/-vy)) 
    else:
        print 'bugs here'
    
    if (360.0 - siftangle) > angle:
        rangle = (360.0 - siftangle) - angle        
    else:
        rangle = 360 - (angle - (360.0 - siftangle))
    return rangle

def dictCode(a,b,max=180):
    return (int(a)/2+1)*max-(max-(int(b)/2+1))
