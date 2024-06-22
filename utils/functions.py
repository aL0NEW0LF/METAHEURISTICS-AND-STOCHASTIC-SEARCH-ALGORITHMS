import numpy as np

def GetFunctionsDetails(F):
    switcher = {
        'F1': (F1, -100, 100, 30),
        'F2': (F2, -10, 10, 30),
        'F3': (F3, -100, 100, 30),
        'F4': (F4, -100, 100, 30),
        'F5': (F5, -30, 30, 30),
        'F6': (F6, -100, 100, 30),
        'F7': (F7, -1.28, 1.28, 30),
        'F8': (F8, -500, 500, 30),
        'F9': (F9, -5.12, 5.12, 30),
        'F10': (F10, -32, 32, 30),
        'F11': (F11, -600, 600, 30),
        'F12': (F12, -50, 50, 30),
        'F13': (F13, -50, 50, 30),
        'F14': (F14, -65.536, 65.536, 2),
        'F15': (F15, -5, 5, 4),
        'F16': (F16, -5, 5, 2),
        'F17': (F17, [-5,0], [10,15], 2),
        'F18': (F18, -2, 2, 2),
        'F19': (F19, 0, 1, 3),
        'F20': (F20, 0, 1, 6),
        'F21': (F21, 0, 10, 4),
        'F22': (F22, 0, 10, 4),
        'F23': (F23, 0, 10, 4),
        'F24': (F24, -100, 100, 30),
        'F25': (F25, -100, 100, 30),
        'F26': (F26, -100, 100, 30),
        'F27': (F27, -100, 100, 30),
        'F28': (F28, -100, 100, 30),
        'F29': (F29, -100, 100, 30),
        'F30': (F30, -100, 100, 30),
        'F31': (F31, -100, 100, 30),
        'F32': (F32, -100, 100, 30),
        'F33': (F33, -100, 100, 30),
        'F34': (F34, -500, 500, 30),
        'F35': (F35, -100, 100, 30),
        'F36': (F36, -100, 100, 30),
        'F37': (F37, -100, 100, 30)
    }
    return switcher.get(F, "Invalid function")

def F1(x):
    return np.sum(np.array(x)**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    dim = len(x)
    return np.sum([np.sum(x[:i+1])**2 for i in range(dim)])

def F4(x):
    return np.max(np.abs(x))

def F5(x):
    dim = len(x)
    x = np.array(x)
    return np.sum(100*(x[1:dim]-(x[0:dim-1]**2))**2 + (x[0:dim-1]-1)**2)

def F6(x):
    x = np.array(x)
    return np.sum(np.abs((x+.5))**2)

def F7(x):
    dim = len(x)
    return np.sum([(i+1)*(x[i]**4) for i in range(dim)]) + np.random.rand()

def F8(x):
    x = np.array(x)
    return np.sum(-x*np.sin(np.sqrt(np.abs(x))))

def F9(x):
    dim = len(x)
    x = np.array(x)
    return np.sum(np.array(x)**2 - 10*np.cos(2*np.pi*x)) + 10*dim

def F10(x):
    dim = len(x)
    x = np.array(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) - np.exp(np.sum(np.cos(2*np.pi*x))/dim) + 20 + np.exp(1)

def F11(x):
    dim = len(x)
    x = np.array(x)
    return np.sum(np.array(x)**2)/4000 - np.prod(np.cos(x/np.sqrt([i+1 for i in range(dim)]))) + 1

def F12(x):
    dim = len(x)
    x = np.array(x)
    return (np.pi/dim)*(10*((np.sin(np.pi*(1+(x[0]+1)/4)))**2) + np.sum((((x[0:dim-1]+1)/4)**2)*(1+10*((np.sin(np.pi*(1+(x[1:dim]+1)/4))))**2) + ((x[dim-1]+1)/4)**2) )+ np.sum(Ufun(x,10,100,4))

def F13(x):
    dim = len(x)
    x = np.array(x)
    return 0.1*((np.sin(3*np.pi*x[0]))**2 + np.sum((x[0:dim-1]-1)**2*(1+(np.sin(3*np.pi*x[1:dim]))**2)) + ((x[dim-1]-1)**2)*(1+(np.sin(2*np.pi*x[dim-1]))**2)) + np.sum(Ufun(x,5,100,4))

def F14(x):
    x = np.array(x)
    aS = np.array([[-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
                   [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]])
    bS = [np.sum((x-aS[:,j])**6) for j in range(25)]
    return (1/500+np.sum(1/np.array([i+1 for i in range(25)]+bS)))**(-1)

def F15(x):
    x = np.array(x)
    aK = [0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    bK = 1/np.array(bK)
    return np.sum((aK-((x[0]*(bK**2+x[1]*bK))/(bK**2+x[2]*bK+x[3])))**2)

def F16(x):
    x = np.array(x)
    return 4*(x[0]**2)-2.1*(x[0]**4)+(x[0]**6)/3+x[0]*x[1]-4*(x[1]**2)+4*(x[1]**4)

def F17(x):
    x = np.array(x)
    return (x[1]-(x[0]**2)*5.1/(4*(np.pi**2))+5/np.pi*x[0]-6)**2+10*(1-1/(8*np.pi))*np.cos(x[0])+10

def F18(x):
    x = np.array(x)
    return (1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*(x[0]**2)-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*(x[0]**2)+48*x[1]-36*x[0]*x[1]+27*(x[1]**2)))

def F19(x):
    x = np.array(x)
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = [1, 1.2, 3, 3.2]
    pH = np.array([[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]])
    return np.sum([cH[i]*np.exp(-(np.sum(aH[i,:]*((x-pH[i,:])**2)))) for i in range(4)])

def F20(x):
    x = np.array(x)
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991], [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    return np.sum([-cH[i]*np.exp(-np.sum(aH[i,:]*((x-pH[i,:])**2))) for i in range(4)])

def F21(x):
    x = np.array(x)
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4])
    return np.sum([-((x-aSH[i,:])@(x-aSH[i,:])+cSH[i])**(-1) for i in range(5)])

def F22(x):
    x = np.array(x)
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3])
    return np.sum([-((x-aSH[i,:])@(x-aSH[i,:])+cSH[i])**(-1) for i in range(7)])

def F23(x):
    x = np.array(x)
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1], [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    return np.sum([-((x-aSH[i,:])@(x-aSH[i,:])+cSH[i])**(-1) for i in range(10)])

def Ufun(x, a, k, m):
    return k*((x-a)**m)*(x>a) + k*((-x-a)**m)*(x<(-a))

# CEC’20 Benchmark funtions
## Bent Cigar Function
def F24(x):
    x = np.array(x)
    return x[0]**2 + 10**6 * np.sum(x[1:]**2)

## Rastrigin’s Function
def F25(x):
    x = np.array(x)
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10)

## High Conditioned Elliptic Function
def F26(x):
    x = np.array(x)
    return np.sum(((10 ** 6) ** (np.arange(len(x)) / (len(x) - 1))) * x ** 2)

## HGBat Function
def F27(x):
    x = np.array(x)
    return ((np.sum(x ** 2)) ** 2 - np.sum(x) ** 2) ** (1 / 2) + (0.5 * np.sum(x ** 2) + np.sum(x)) / len(x) + 0.5

## Rosenbrock’s Function
def F28(x):
    x = np.array(x)
    return np.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (x[:-1] - 1) ** 2)

## Griewank’s Function
def F29(x):
    x = np.array(x)
    return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

## Ackley’s Function
def F30(x):
    x = np.array(x)
    return -20 * np.exp(-0.2 * ((np.sum(x ** 2) / len(x)) ** 0.5)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / len(x)) + 20 + np.exp(1)

## Happycat Function
def F31(x):
    x = np.array(x)
    return (np.absolute(np.sum(x ** 2) - len(x))) ** (1 / 4) + (0.5 * np.sum(x ** 2) + np.sum(x)) / len(x) + 0.5

## Discus Function
def F32(x):
    x = np.array(x)
    return (10 ** 6) * (x[0] ** 2) + np.sum(x[1:] ** 2)

## Lunacek bi-Rastrigin Function
def F33(x):
    x = np.array(x)
    D = len(x)
    d = 1
    s = 1 - 1 / (2 * ((D + 20) ** (1 / 2)) - 8.2)
    mu1 = 2.5
    mu2 = -np.sqrt((mu1 ** 2 - d) / s)
    return np.min(np.sum((x - mu1) ** 2, axis=1), d * D + s * np.sum((x - mu2) ** 2, axis=1)) + 10 * (D - np.sum(np.cos(2 * np.pi * (x - mu1))))

## Modified Schwefel’s Function
def F34(x):
    def g(z):
        if np.absolute(z) <= 500:
            return z * np.sin((np.absolute(z)) ** (1 / 2))
        elif z > 500:
            return (500 - z % 500) * np.sin((500 - z % 500) ** (1 / 2)) - (z - 500) ** 2 / (10000 * len(x))
        elif z < -500:
            return (np.absolute(z) % 500 - 500) * np.sin(np.absolute(np.absolute(z) % 500 - 500) ** (1 / 2)) - (z + 500) ** 2 / (10000 * len(x)) 
        
    x = np.array(x)
    z = x + 4.209687462275036e+002

    return 418.9829 * len(x) - np.sum(g(z))

## Expanded Schaffer’s Function
def F35(x):
    ### Schaffer’s Function
    def g(x, y):
        return 0.5 + ((np.sin(x ** 2 - y ** 2) ** 2 - 0.5) / (1 + 0.001 * (x ** 2 + y ** 2)) ** 2)
    
    x = np.array(x)
    return np.sum(g(x[:-1], x[1:])) + g(x[-1], x[0])

## Expanded Rosenbrock’s plus Griewangk’s Function
def F36(x):
    x = np.array(x)
    return F29(F28(x)) + (100 * (x[-1] ** 2 - x[0]) ** 2 + (x[-1] - 1) ** 2)

## Weierstrass Function
def F37(x):
    x = np.array(x)
    a = 0.5
    b = 3
    kmax = 20
    return np.sum([np.sum([(a ** k) * np.cos(2 * np.pi * (b ** k) * (x + 0.5)) for k in range(kmax)]) - len(x) * np.sum([(a ** k) * np.cos(2 * np.pi * (b ** k) * 0.5) for k in range(kmax)])])   