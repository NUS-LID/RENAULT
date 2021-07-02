import torch
import math

def p_i(mu, sigma, li, wi, a, b):
    def fn(x):
        return torch.erf( (x-mu)/(math.sqrt(2)*sigma) )
    Z = 1/2 * (fn(b) - fn(a))
    pi = 1/(2*Z) * (fn(li+wi) - fn(li))    
    return pi

def p_categorical(mu, sigma, a, b, n):
    w = (b-a)/n
    return torch.stack([p_i(mu, sigma, a+i*w, w, a, b) 
                        for i in range(n)]).permute(1,0)

def create_p_categorical(a, b, n, ratio=0.5, sigma=None, debug=False):
    if sigma is None:
        sigma = ratio * ((b-a)/n)
    if debug:
        ratio = sigma / ((b-a)/n)
        print('sigma:', sigma,'| ratio:', ratio)
    return lambda mu: p_categorical(mu, sigma, a, b, n)


if __name__ == '__main__':
    reward_categorical = create_p_categorical(a=-1, b=1, n=21, debug=True)
    activity_categorical = create_p_categorical(a=0, b=255, n=51, debug=True)
    activity = activity_categorical(torch.tensor([128]))
    print(activity)
    reward = reward_categorical(torch.tensor([0]))
    print(reward)