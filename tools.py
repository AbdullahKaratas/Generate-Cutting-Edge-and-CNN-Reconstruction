import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pandas as pd
import warnings


def dxdy(name):
    if name == 'sample_typeA.txt':
        dx = 1e-06
        dy = 1e-06
        return dx, dy

    if name == 'sample_typeB.txt':
        dx = 4e-07
        dy = 4-07
        return dx, dy


def maxpointsXYZ(X, Y, Z):
    numberofrows, numberofcols = np.shape(Z)
    maxofXYZ = np.zeros((3, numberofrows))
    amaxv = np.zeros(numberofrows)
    for i in range(0, numberofrows):
        ztemp = Z[i, :]
        TestIfTrue = np.sum(np.isnan(ztemp))
        if TestIfTrue < numberofcols:
            amaxztemp = np.nanargmax(ztemp)
            maxofXYZ[0, i] = X[i, amaxztemp]
            maxofXYZ[1, i] = Y[i, 0]
            maxofXYZ[2, i] = Z[i, amaxztemp]
            amaxv[i] = amaxztemp
        else:
            maxofXYZ[0, i] = np.NaN
            maxofXYZ[1, i] = np.NaN
            maxofXYZ[2, i] = np.NaN

    return maxofXYZ, amaxv  # Argumengt of Max Points and Values of Max Points


def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(round(window_len/2)-1):-(round(window_len/2))]


def derivativ(x, y, iMax=-1):
    dy = np.zeros(y.shape, np.float)
    dy[0:iMax-1] = np.diff(y[0:iMax])/np.diff(x[0:iMax])
    dy[iMax] = (y[iMax] - y[-2])/(x[iMax] - x[-2])
    return dy


def generatex(dx, numberofrows, numberofcols):
    Xcol = np.zeros((1, numberofcols))
    for i in range(0, numberofcols):
        Xcol[0, i] = i*dx

    X = np.ones((numberofrows, 1)) @ Xcol
    return X  # Output X Matrix


def generatey(dy, numberofrows, numberofcols):
    Yrow = np.zeros((numberofrows, 1))
    for i in range(0, numberofrows):
        Yrow[i, 0] = i*dy

    Y = Yrow @ np.ones((1, numberofcols))
    return Y  # Output Y Matrix


def smoothingdata(maxofXYZ, t, sv):
    t = t.T
    x = maxofXYZ[0, :]
    y = maxofXYZ[1, :]
    z = maxofXYZ[2, :]
    sx = UnivariateSpline(t, x, s=sv)
    sy = UnivariateSpline(t, y, s=sv)
    sz = UnivariateSpline(t, z, s=sv)

    sx = sx(t)
    sy = sy(t)
    sz = sz(t)

    smoothinmaxXZY = np.block([[sx], [sy], [sz]])

    return sx, sy, sz  # Smoothimng of Max Points along t


def parametrisierung(maxofXYZ):
    numberofrows, numberofcols = np.shape(maxofXYZ)
    t = np.zeros(numberofcols)
    for i in range(1, numberofcols):
        if np.isnan(maxofXYZ[2, i-1]) == False and np.isnan(maxofXYZ[2, i]) == False:
            dx = maxofXYZ[0, i] - maxofXYZ[0, i-1]
            dy = maxofXYZ[1, i] - maxofXYZ[1, i-1]
            dz = maxofXYZ[2, i] - maxofXYZ[2, i-1]
            h = np.sqrt(dx*dx + dy*dy + dz*dz)
            t[i] = t[i-1] + h

    return t  # Output t (for Spline later)


def generatePoints(ddzTemp, dzTemp, zTemp, iMax, nrBins=10):

    if nrBins > 12:
        nrBins = 12
        warnings.warn('nrBins was set to 12')
    elif nrBins < 5:
        nrBins = 5
        warnings.warn('nrBins was set to 5')

    point = []
    val1 = next((i for i, x in enumerate(
        ddzTemp[iMax:iMax-50:-1]) if x > 0), None)
    point.append(int(np.around(iMax - val1, decimals=0)))

    if len(point) == 0:
        leftToStart = int(np.around(iMax, decimals=0))
    else:
        leftToStart = int(np.around(iMax - val1, decimals=0))

    val2 = next((i for i, x in enumerate(
        ddzTemp[iMax:iMax+50]) if x > 0), None)
    point.append(int(np.around(iMax + val2, decimals=0)))

    if len(point) == 0 or len(point) == 1:
        rightToEnd = int(np.around(iMax, decimals=0))
    else:
        rightToEnd = int(np.around(iMax + val2, decimals=0))

    histo, binEdges = np.histogram(dzTemp[0:leftToStart], bins=nrBins)
    binSize = binEdges[2] - binEdges[1]
    # Zum schluss sind die größten werte (argumente)
    argSortHisto = np.argsort(histo)
    # -3 sieht mir sehr willkürlich aus ...
    histogr = pd.DataFrame(
        {'orig_index': argSortHisto[-5:], 'hist': histo[argSortHisto[-5:]], 'binEdges': binEdges[argSortHisto[-5:]]})
    histogr = histogr.sort_values(by='hist', ascending=False)

    point.append(int(np.around(0, decimals=0)))
    findRange = np.array(np.where(np.logical_and(
        histogr['binEdges'][4] <= dzTemp[0:leftToStart], dzTemp[0:leftToStart] <= histogr['binEdges'][4] + binSize)))
    if leftToStart - findRange[0, -1] > 20 and findRange[0, -1] > 20:
        point.append(int(np.around(findRange[0, -1], decimals=0)))

    if histogr['hist'][3] / histogr['hist'][4] > 0.3:
        findRange0 = np.array(np.where(np.logical_and(
            histogr['binEdges'][3] <= dzTemp[0:leftToStart], dzTemp[0:leftToStart] <= histogr['binEdges'][3] + binSize)))
        if leftToStart - findRange0[0, -1] > 20 and findRange0[0, -1] > 20:
            point.append(int(np.around(findRange0[0, -1], decimals=0)))

    if histogr['hist'][2] / histogr['hist'][4] > 0.3:
        findRange1 = np.array(np.where(np.logical_and(
            histogr['binEdges'][2] <= dzTemp[0:leftToStart], dzTemp[0:leftToStart] <= histogr['binEdges'][2] + binSize)))
        if leftToStart - findRange1[0, -1] > 20 and findRange1[0, -1] > 20:
            point.append(int(np.around(findRange1[0, -1], decimals=0)))

    if histogr['hist'][1] / histogr['hist'][4] > 0.3:
        findRange2 = np.array(np.where(np.logical_and(
            histogr['binEdges'][1] <= dzTemp[0:leftToStart], dzTemp[0:leftToStart] <= histogr['binEdges'][1] + binSize)))
        if leftToStart - findRange2[0, -1] > 20 and findRange2[0, -1] > 20:
            point.append(int(np.around(findRange2[0, -1], decimals=0)))

    if histogr['hist'][0] / histogr['hist'][4] > 0.3:
        findRange3 = np.array(np.where(np.logical_and(
            histogr['binEdges'][0] <= dzTemp[0:leftToStart], dzTemp[0:leftToStart] <= histogr['binEdges'][0] + binSize)))
        if leftToStart - findRange3[0, -1] > 20 and findRange3[0, -1] > 20:
            point.append(int(np.around(findRange3[0, -1], decimals=0)))

    histo, binEdges = np.histogram(dzTemp[rightToEnd:-1], bins=nrBins)
    binSize = binEdges[2] - binEdges[1]
    argSortHisto = np.argsort(histo)
    histogr = pd.DataFrame(
        {'orig_index': argSortHisto[-5:], 'hist': histo[argSortHisto[-5:]], 'binEdges': binEdges[argSortHisto[-5:]]})
    histogr = histogr.sort_values(by='hist', ascending=False)

    findRange = np.array(np.where(np.logical_and(
        histogr['binEdges'][4] <= dzTemp[rightToEnd:-1], dzTemp[rightToEnd:-1] <= histogr['binEdges'][4] + binSize)))
    if findRange[0, -1] > 20:
        point.append(int(np.around(findRange[0, -1] + rightToEnd, decimals=0)))

    if histogr['hist'][3] / histogr['hist'][4] > 0.2 and np.abs(histogr['binEdges'][3] - histogr['binEdges'][4]) > 2 * binSize:
        findRange0 = np.array(np.where(np.logical_and(
            histogr['binEdges'][3] <= dzTemp[rightToEnd:-1], dzTemp[rightToEnd:-1] <= histogr['binEdges'][3] + binSize)))
        if findRange0[0, -1] > 20 and zTemp.shape[0]-1 - findRange0[0, -1] - rightToEnd > 20:
            point.append(
                int(np.around(findRange0[0, -1] + rightToEnd, decimals=0)))

    if histogr['hist'][2] / histogr['hist'][4] > 0.2 and np.abs(histogr['binEdges'][2] - histogr['binEdges'][4]) > 2 * binSize and np.abs(histogr['binEdges'][2] - histogr['binEdges'][3]) > 2 * binSize:
        findRange1 = np.array(np.where(np.logical_and(
            histogr['binEdges'][2] <= dzTemp[rightToEnd:-1], dzTemp[rightToEnd:-1] <= histogr['binEdges'][2] + binSize)))
        if findRange1[0, -1] > 20 and zTemp.shape[0]-1 - findRange1[0, -1] - rightToEnd > 20:
            point.append(
                int(np.around(findRange1[0, -1] + rightToEnd, decimals=0)))

    if histogr['hist'][1] / histogr['hist'][4] > 0.2 and np.abs(histogr['binEdges'][1] - histogr['binEdges'][4]) > 2 * binSize and np.abs(histogr['binEdges'][1] - histogr['binEdges'][3]) > 2 * binSize and np.abs(histogr['binEdges'][1] - histogr['binEdges'][2]) > 2 * binSize:
        findRange2 = np.array(np.where(np.logical_and(
            histogr['binEdges'][1] <= dzTemp[rightToEnd:-1], dzTemp[rightToEnd:-1] <= histogr['binEdges'][1] + binSize)))
        if findRange2[0, -1] > 20 and zTemp.shape[0]-1 - findRange2[0, -1] - rightToEnd > 20:
            point.append(
                int(np.around(findRange2[0, -1] + rightToEnd, decimals=0)))

    if histogr['hist'][0] / histogr['hist'][4] > 0.2 and np.abs(histogr['binEdges'][0] - histogr['binEdges'][4]) > 2 * binSize and np.abs(histogr['binEdges'][0] - histogr['binEdges'][3]) > 2 * binSize and np.abs(histogr['binEdges'][0] - histogr['binEdges'][2]) > 2 * binSize and np.abs(histogr['binEdges'][0] - histogr['binEdges'][1]) > 2 * binSize:
        findRange3 = np.array(np.where(np.logical_and(
            histogr['binEdges'][0] <= dzTemp[rightToEnd:-1], dzTemp[rightToEnd:-1] <= histogr['binEdges'][0] + binSize)))
        if findRange3[0, -1] > 20 and zTemp.shape[0]-1 - findRange3[0, -1] - rightToEnd > 20:
            point.append(
                int(np.around(findRange3[0, -1] + rightToEnd, decimals=0)))

    point.append(int(np.around(zTemp.shape[0]-1, decimals=0)))
    point.sort()

    # Small optimization:
    # If a point is found, search for a local minimum:
    point = np.array(point)
    var1 = np.array(np.where(np.logical_and(point > 0, point < leftToStart)))
    if var1.shape[1] == 1:
        case11ArgMin = np.argmin(ddzTemp[0:point[var1[0, 0]]])
        case12ArgMin = np.argmin(ddzTemp[point[var1[0, 0]]:leftToStart])
        case11ValMin = np.min(ddzTemp[0:point[var1[0, 0]]])
        case12ValMin = np.min(ddzTemp[point[var1[0, 0]]:leftToStart])
        if case11ValMin < case12ValMin:
            case1 = case11ArgMin
            point[var1[0, 0]] = case1
        else:
            case1 = case12ArgMin
            point[var1[0, 0]] = case1 + point[var1[0, 0]]

    var2 = np.array(np.where(np.logical_and(
        point > rightToEnd, point < zTemp.shape[0]-1)))
    if var2.shape[1] == 1:
        case21ArgMin = np.argmin(ddzTemp[rightToEnd:point[var2[0, 0]]])
        case22ArgMin = np.argmin(ddzTemp[point[var2[0, 0]]:-1])
        case21ValMin = np.min(ddzTemp[rightToEnd:point[var2[0, 0]]])
        case22ValMin = np.min(ddzTemp[point[var2[0, 0]]:-1])
        if case21ValMin < case22ValMin:
            case2 = case21ArgMin
            point[var2[0, 0]] = case2 + rightToEnd
        else:
            case2 = case22ArgMin
            point[var2[0, 0]] = case2 + point[var2[0, 0]]

    return point


def polyfit_with_fixed_points(d, x, y, xf, yf):
    # Source from https://stackoverflow.com/questions/15191088/how-to-do-a-polynomial-fit-with-fixed-points
    mat = np.empty((d + 1 + len(xf),) * 2)
    vec = np.empty((d + 1 + len(xf),))
    x_n = x**np.arange(2 * d + 1)[:, None]
    yx_n = np.sum(x_n[:d + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(d + 1) + np.arange(d + 1)[:, None]
    mat[:d + 1, :d + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(d + 1)[:, None]
    mat[:d + 1, d + 1:] = xf_n / 2
    mat[d + 1:, :d + 1] = xf_n.T
    mat[d + 1:, d + 1:] = 0
    vec[:d + 1] = yx_n
    vec[d + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:d + 1]


def getParams(point, iMax, xTemp, zTemp, d=3, vary='yes'):
    leftOfImax = np.array(np.where(point < iMax))
    leftFromIMax = point[leftOfImax[0, -1]]
    zSim = np.array(zTemp[point[leftOfImax[0, 0]]])
    params = np.zeros((1, d+1))
    for i in range(0, leftOfImax[0, :].shape[0]-1):
        xf = np.array([xTemp[point[leftOfImax[0, i]]],
                       xTemp[point[leftOfImax[0, i+1]]]])
        zf = np.array([zTemp[point[leftOfImax[0, i]]],
                       zTemp[point[leftOfImax[0, i+1]]]])
        parameter = polyfit_with_fixed_points(
            d, xTemp[point[leftOfImax[0, i]]:point[leftOfImax[0, i+1]]], zTemp[point[leftOfImax[0, i]]:point[leftOfImax[0, i+1]]], xf, zf)

        # Change Parameter here
        if vary == 'yes':
            for ak in range(0, len(parameter)):
                parameter[ak] = np.random.uniform(
                    0.99985 * parameter[ak], (2-0.99985) * parameter[ak])

        poly = np.polynomial.Polynomial(parameter)
        sizeOfPoints = point[leftOfImax[0, i+1]] - point[leftOfImax[0, i]] + 1
        xx = np.linspace(xTemp[point[leftOfImax[0, i]]],
                         xTemp[point[leftOfImax[0, i+1]]], sizeOfPoints)
        zSim = np.hstack((zSim, poly(xx[1:])))
        params = np.vstack((params, parameter))

    # Middle
    rightOfImax = np.array(np.where(point > iMax))
    rightFromIMax = point[rightOfImax[0, 0]]
    xf = np.array([xTemp[leftFromIMax], xTemp[iMax], xTemp[rightFromIMax]])
    zf = np.array([zTemp[leftFromIMax], zTemp[iMax], zTemp[rightFromIMax]])
    parameter = polyfit_with_fixed_points(
        d, xTemp[leftFromIMax:rightFromIMax], zTemp[leftFromIMax:rightFromIMax], xf, zf)

    if vary == 'yes':
        for ak in range(0, len(parameter)):
            parameter[ak] = np.random.uniform(
                0.99985 * parameter[ak], (2-0.99985) * parameter[ak])

    poly = np.polynomial.Polynomial(parameter)
    sizeOfPoints = rightFromIMax - leftFromIMax + 1
    xx = np.linspace(xTemp[leftFromIMax], xTemp[rightFromIMax], sizeOfPoints)
    zSim = np.hstack((zSim, poly(xx[1:])))
    params = np.vstack((params, parameter))

    # Right
    for i in range(0, rightOfImax[0, :].shape[0]-1):
        xf = np.array([xTemp[point[rightOfImax[0, i]]],
                       xTemp[point[rightOfImax[0, i+1]]]])
        zf = np.array([zTemp[point[rightOfImax[0, i]]],
                       zTemp[point[rightOfImax[0, i+1]]]])
        parameter = polyfit_with_fixed_points(
            d, xTemp[point[rightOfImax[0, i]]:point[rightOfImax[0, i+1]]], zTemp[point[rightOfImax[0, i]]:point[rightOfImax[0, i+1]]], xf, zf)

        if vary == 'yes':
            for ak in range(0, len(parameter)):
                parameter[ak] = np.random.uniform(
                    0.99995 * parameter[ak], (2-0.99995) * parameter[ak])

        poly = np.polynomial.Polynomial(parameter)
        sizeOfPoints = point[rightOfImax[0, i+1]] - \
            point[rightOfImax[0, i]] + 1
        xx = np.linspace(xTemp[point[rightOfImax[0, i]]],
                         xTemp[point[rightOfImax[0, i+1]]], sizeOfPoints)
        zSim = np.hstack((zSim, poly(xx[1:])))
        params = np.vstack((params, parameter))

    return params[1:], zSim


def getZSim(point, iMax, xTemp, zTemp, d=3, sv=0.005, vary='yes'):
    # Left
    parameter, zSim = getParams(point, iMax, xTemp, zTemp, d=3, vary=vary)

    zSim = UnivariateSpline(xTemp, zSim, s=sv)
    zSim = zSim(xTemp)

    return zSim


def smoothZSim(YSIM, ZSIM, sv=0.005):
    # We smooth in E_{yz}
    totalRun = ZSIM.shape[1]
    for j in range(0, totalRun):
        zTemp = ZSIM[:, j]
        yTemp = YSIM[:, j]

        ts = np.argwhere(np.isnan(zTemp) == 0)
        ts = ts.ravel()
        yTemp = yTemp[ts]
        zTemp = zTemp[ts]

        if zTemp.shape[0] >= 5:
            zSim = UnivariateSpline(yTemp, zTemp, s=sv)
            ZSIM[ts, j] = zSim(yTemp)

    return ZSIM


def generateDefect(point, iMax, xTemp, indexLength, depth, dy, XNEW, sx, argmaxOfSimulatedProfil, startDefect=0.3, maxWidthDefect=0.375, minWidthDefect=0.25, shift='off'):
    # zDepth

    if shift == 'off':
        vShift = 0
    else:
        vShift = int(np.random.uniform(low=-40, high=40))

    print(vShift)
    startInY = int(startDefect * XNEW.shape[0])
    endInY = int(startDefect * XNEW.shape[0]) + indexLength
    y = np.array([startInY*dy, 0.5*(startInY + endInY)*dy, endInY*dy])
    z = np.array([0, depth, 0])
    zCoef = np.polyfit(y, z, 2)
    p = np.poly1d(zCoef)
    yy = np.linspace(0, indexLength*dy, indexLength)
    zDepth = p(yy)

    iMax = iMax

    leftOfImax = np.array(np.where(point < iMax))
    leftFromIMax = point[leftOfImax[0, -1]]
    rightOfImax = np.array(np.where(point > iMax))
    rightFromIMax = point[rightOfImax[0, 0]]
    sizeOfPoints = rightFromIMax - leftFromIMax + 1

    # LeftEdge
    y = np.array([0, int(0.5*indexLength), indexLength])
    z = np.array([int(leftFromIMax + sizeOfPoints * maxWidthDefect), int(leftFromIMax +
                                                                         sizeOfPoints * minWidthDefect), int(leftFromIMax + sizeOfPoints * maxWidthDefect)])
    zCoef = np.polyfit(y, z, 2)
    p = np.poly1d(zCoef)
    yy = np.linspace(0, indexLength - 1, indexLength)
    leftDefectEdge = p(yy)

    # RightEdge
    y = np.array([0, int(0.5*indexLength), indexLength])
    z = np.array([int(rightFromIMax - sizeOfPoints * maxWidthDefect), int(rightFromIMax -
                                                                          sizeOfPoints * minWidthDefect), int(rightFromIMax - sizeOfPoints * maxWidthDefect)])
    zCoef = np.polyfit(y, z, 2)
    p = np.poly1d(zCoef)
    yy = np.linspace(0, indexLength - 1, indexLength)
    rightDefectEdge = p(yy)

    Defect = np.zeros(XNEW.shape)
    k = 0
    for i in range(int(startDefect * XNEW.shape[0]), int(startDefect * XNEW.shape[0]) + indexLength):
        xTempAll = XNEW[i, :]
        iMaxTemp = np.argmin(abs(xTempAll - sx[i]))
        deltaInX = iMaxTemp - argmaxOfSimulatedProfil
        leftEdge = int(leftDefectEdge[k])
        rightEdge = int(rightDefectEdge[k])
        middle = int(0.5 * (leftEdge + rightEdge))
        depthTemp = zDepth[k]
        x = np.array([xTemp[leftEdge], xTemp[middle], xTemp[rightEdge]])
        z = np.array([0, depthTemp, 0])
        zCoef = np.polyfit(x, z, 2)
        p = np.poly1d(zCoef)
        sizeOfPoints = -leftEdge + rightEdge + 1
        xx = np.linspace(xTemp[leftEdge], xTemp[rightEdge], sizeOfPoints)
        zz = p(xx)
        zz = zz * np.random.rand(zz.shape[0]) * 1.5
        Defect[i, vShift+deltaInX+leftEdge:vShift+deltaInX+rightEdge+1] = zz
        k = k + 1

    return Defect
