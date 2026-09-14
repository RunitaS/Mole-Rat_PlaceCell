# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:25:12 2022

@author: ADITI

Spike Train Analysis
Inter-Spike Interval (ISI) distribution - getISI: computes ISIs, coefficient of variation (CV), and plots ISI histogram
Firing rate over time - getFR: calculates firing rate using a sliding window, plus the Fano Factor (FF) for spike count variability
Firing rate phase plane - getFRphaseplane: computes smoothed firing rates for two cell types (e.g., place cell vs. interneuron) over time
Temporal autocorrelogram - TempAutocorr: builds an autocorrelogram from continuous spike times (+-1000 ms window, 20 ms bins)
Intrinsic oscillation frequency - precessionFreq: estimates the cell's intrinsic firing frequency from the autocorrelogram peak

Theta Phase Analysis
Spike theta phase estimation - getPhase: assigns a theta phase to each spike by linear interpolation between LFP peaks
Theta Modulation Index (TMI) - calcTMI: quantifies how strongly a cell's firing is modulated by theta phase
Moving average of phase vs. position - movingavgPhase: circular moving average of spike phase along position
Preferred phase of interneuron - findprefPhaseInh: finds the preferred inhibitory phase outside the place field
Phase valley detection - findPhaseValley: finds the trough in the spike phase distribution (useful for setting precession cycle boundaries)
Theta skip amplitude - thetaskipAmp: quantifies theta-skipping by comparing the 1st and 2nd autocorrelogram peaks (~125 ms vs. ~250 ms)

Phase Precession Analysis
Linear regression for precession - linRegPrecession: finds the best-fit negative-slope linear regression across all 360deg phase windows
Standard linear fit - linfit: ordinary least-squares fit of position vs. phase, returns slope, intercept, R^2, and deviation from fit
Circular-linear regression - circRegress / lcc: fits a circular-linear model to phase vs. position using slope optimization (Kempter 2012 method)
Kempter circular correlation - kempCorr: computes circular-circular correlation between circularized position and spike phase
Spike density plot - plotSpikeDensity: 2D histogram of spike position vs. phase, Gaussian-smoothed
Precession scatter plot - plotPrecession: scatter plot of position vs. phase over two theta cycles

Place Field Analysis
Rate map construction - frRatemap: builds a 1D firing rate map (raw + Gaussian-smoothed) from spike positions and occupancy
Place field detection - fieldDetect: isolates spikes within the place field using a 10% firing rate threshold; returns field boundaries, centre, and normalized positions
Spatial information score - spinfo: computes Skaggs spatial information (bits/spike) from the rate map
Mehta skewness - calcMehtaSkew: calculates the skewness of the place field firing rate distribution (Mehta 2000 method) - positive skew indicates forward asymmetry
FFT power ratio of rate map - fftPowerRatio: frequency analysis of the rate map to quantify low vs. high spatial frequency content

Contour / Cluster Analysis (on 2D phase-position data)
Contour-based cluster isolation - isolateContours: finds the dominant cluster in position-phase space using density contours
Ellipse fitting on clusters - part of isolateContours: fits an ellipse to the isolated cluster, returning centre, major/minor radii, and rotation angle
Polygon containment - calcWithinPolygon: identifies which spikes fall inside a user-defined polygon region
Contour plotting - plotContour: visualizes probability-level contours over a 2D density map
GMM clustering - GMMcluster: fits a Gaussian Mixture Model to spike phase data across multiple theta cycles

Curve Fitting & Statistics
Quadratic fit - quadfit: fits a quadratic curve to data, returns R^2 and coefficients
Linear curve fit - lineqnfit: fits a line using curve_fit, returns R^2, slope, intercept
Moving window computation - movingwinCompute: applies mean, median, or linear slope in a sliding window over position
Moving window derivative - movingDerivative: estimates local rate of change in a sliding window
Numerical derivative - derivative: computes nth-order finite differences
Skew-normal input statistics - spatialInputstats: returns mean, variance, skewness, and kurtosis of a skew-normal distribution

Utilities
Gaussian smoothing - gaussSmooth: 1D convolution with a Gaussian kernel (NaN-preserving)
NaN / Inf removal - removeNaNs, removeinfs: cleans arrays before analysis
Continuous spike time reconstruction - continuousSpktimes: concatenates per-lap spike times into a single continuous time series
E/I ratio - calcEIratio: computes excitation/inhibition ratio and net excitation from total synaptic currents
"""

import numpy as np
from scipy.signal import find_peaks
from collections import Counter
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import pandas as pd
import scipy as sp
from scipy.stats import skewnorm
import warnings
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.stats import skew
from math import *
from numpy.fft import fft
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def testsum(a,b):
    return(a+b)

def removeNaNs(x,y):
    # remove nans from data:
    nanindy = np.where(np.isnan(y))
    nanindx = np.where(np.isnan(x))
    nanind = np.unique(np.append(nanindy,nanindx))
    y = np.delete(y,nanind)
    x = np.delete(x,nanind)

    return x,y

def removeinfs(x,y):
    # remove nans from data:
    infindy = np.where(np.isinf(y))
    infindx = np.where(np.isinf(x))
    infind = np.unique(np.append(infindy,infindx))
    y = np.delete(y,infind)
    x = np.delete(x,infind)

    return x,y

# function to return different spike parameters based on spike timestamps
def getISI(spikets,plotit=False,binwidth=5):
    # returns isi in ms
    # check if any nan values in the array
    if any(np.isnan(spikets)):
        warnings.warn('CV will not be reliable as there are nans present in the data')
    isi = np.diff(spikets)
    #check if the spikets are in seconds
    if any((isi>0) & (isi<0.1)):
        isi= isi*1000 # convert to ms
    # coefficient of variation:
    CV = np.std(isi)/np.mean(isi)

    if plotit==True:
        plt.figure()
        plt.hist(isi,bins= np.arange(min(isi),max(isi)+binwidth, binwidth))
        plt.xlabel('isi (ms)')
        plt.ylabel('counts')
        plt.title('CV = ' + str(round(CV,2)))

    return isi,CV


def gauss(n=11,sigma=1):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


def gaussSmooth(y,n=11,sigma=1,returnnan=True):
    g = gauss(n=n,sigma=sigma)
    #dealing with nan values
    if np.isnan(y).any():
        nanid = np.where(np.isnan(y))
        y = np.nan_to_num(y)
    yy = np.pad(y,(len(g),), 'edge') # padding like trailing ends is better than padding with 0
    paddedidx = np.arange(0,len(g),1)
    paddedidx = np.append(paddedidx,len(yy)-np.arange(1,len(g)+1,1))
    y2 = np.convolve(yy,g,'same')
    y2=np.delete(y2,paddedidx)

    if returnnan and np.isnan(y).any():
        y2[nanid]=np.nan

    return y2

# calculate firing rate with time in a moving window of 50ms
def getFR(spikets,t,winlen=0.05,overlap=0.5,plotit=False):
    # check if any nan values in the array
    if any(np.isnan(spikets)):
        warnings.warn('CV will not be reliable as there are nans present in the data')

    #check if the spikets are in seconds
    isi = np.diff(spikets)
    if not any((isi>0) & (isi<0.1)):
        isi= isi/1000

    step = winlen*(1-overlap) # moving window
    tbins = np.arange(t[0],t[-1],step)
    fr=[]
    numspk=[]
    for j in tbins:
        win = [j,j+winlen]
        idx = np.where((spikets>win[0])&(spikets<win[1]))
        fr.append(len(spikets[idx])/winlen)
        numspk.append(len(spikets[idx]))
        j= win[1]+step

    # fano factor:
    FF = np.var(numspk)/np.mean(numspk)

    if plotit==True:
        meanfr = np.mean(fr)
        plt.figure()
        plt.plot(tbins,fr)
        plt.ylabel('Hz= numspks / time interval ')
        plt.xlabel('seconds')
        plt.title('FanoFactor= '+str(round(FF,2))+'\n meanFr= '+str(round(meanfr,1))+' Hz')

    return fr,FF,tbins

def getFRphaseplane(spktspc,spktsin,winlen=0.2):
    t = np.arange(0,max(spktsin)+winlen,winlen)
    frpc=[]
    frin=[]
    for j in range(1,len(t)):
        idx = np.where((spktspc>t[j-1])&(spktspc<=t[j]))
        frpc.append(len(idx[0])/winlen)
        idxin = np.where((spktsin>t[j-1])&(spktsin<=t[j]))
        frin.append(len(idxin[0])/winlen)

    # smfrpc = gaussian_filter1d(frpc,2)
    smfrpc = gaussSmooth(frpc,n=7,sigma=2)
    # smfrin = gaussian_filter1d(frin,2)
    smfrin = gaussSmooth(frin,n=7,sigma=2)

    return smfrpc,smfrin

# estimate theta phases by linear interpolation
def getPhase(spkTimes,sinInput,time):
    idx,_ = find_peaks(sinInput)
    pkTimes = time[idx]#*second
    spkPhase = np.zeros(len(spkTimes))

    for i in range(len(spkTimes)):
        if spkTimes[i]>=pkTimes[0] and spkTimes[i]<=pkTimes[-1]:
            pkTsBefore = max(pkTimes[pkTimes<=spkTimes[i]])
            pkTsAfter = min(pkTimes[pkTimes>=spkTimes[i]])
            interpkInt = pkTsAfter-pkTsBefore
            #can include an additional here to check if the inter peak interval is within theta range.
            # it is not necessary here since our inputs are in the theta range only
            spkPhase[i]=((spkTimes[i]-pkTsBefore)/interpkInt)*360
        else:
            spkPhase[i]=np.nan

    return spkPhase


# finding the derivative to get rate of change
def derivative(x,y,order=1):
    if any(np.isnan(x)) or any(np.isnan(y)):
        print('Removing nan from the data for accurate differentiation')
        idx1 = np.array(np.where(np.isnan(x)))
        idx2 = np.array(np.where(np.isnan(y)))
        ix =np.append(idx1,idx2)
        x = np.delete(x,ix)
        y = np.delete(y,ix)

    der = np.diff(y)/np.diff(x) # returns atleast the first derivative

    if order>1:
        for j in range(1,order):
            der_n = der
            der = np.diff(der)/np.diff(x[0:-j])
            if round(np.mean(der))==0: #if the nth derivative is 0
                der=der_n # return the n-1 derivative
                print('Non zero differentiation could be performed only till order= '+str(j))
                break
    xder = x[0:-order]

    # convert the nans creating by 0/0 divide back to 0
    if np.isnan(der).any():
        nanid = np.where(np.isnan(der))
        der[nanid[0][0]]= 0
    if np.isnan(xder).any():
        nanid = np.where(np.isnan(xder))
        xder[nanid[0][0]]=0

    return der, xder


# theta modulation index: (cross check with the matlab code once)
def calcTMI(spkphases, plotit=False, findpeaks = False):
    # make 5 copies of spike tehta phases>> smooth gaussian >> take centre 2 cycles >> max normalize >> plot and calculate TMI
    phscopies = []
    numcycles = 5
    binwidth=36
    for j in range(numcycles):
        phscopies.extend(spkphases+(j*360))
    h=plt.hist(phscopies,bins=np.arange(0,numcycles*360,binwidth))# 50bins is 36 degrees binwidth.
    counts = h[0]
    edges = h[1]

    # using smoothing properties of winsize=7 and sigma =0.5 as in the matlab function calcTMIv6
    s=0.5
    w = 7
    #gaussian_filter1d uses lw  = int(truncate*sigma +0.5) so t becomes
    t = (w -0.5)/s # default value of truncate is 4.
    # smcounts = gaussian_filter1d(counts,s,truncate=t)
    smcounts = gaussSmooth(counts,n=w,sigma=s)
    edges2 = edges[0:-1]
    idx = np.where((edges2>=360)& (edges2<=1080))
    edges2 = edges2[idx]-360
    counts2 = smcounts[idx]
    normcounts = counts2/max(counts2)
    tmi = 1-min(normcounts)

    bincentres = edges2+binwidth/2
    if plotit:
        ax = plotit
        ax.clear() # for hist
        ax.bar(x=edges2,height=normcounts,width = np.diff(np.append(edges2,np.nan)),align='edge',fc='skyblue',ec='black') # check alignment
        ax.set_title('TMI=' +str(np.round(tmi,2)))
        ax.plot(bincentres,normcounts,linewidth=2)
        ax.set_xticks([0,360,720])
        ax.set_xlabel('degrees')

    if findpeaks:
        return tmi, edges2, bincentres, normcounts
    else:
        return tmi

# plot spike density (put this into a function if will be plotting for different combinations)
def plotSpikeDensity(x,y,xedges=np.arange(0,2.02,0.02),yedges=np.arange(0,720,10),gaussSigma=1,cm=True,plotit=True):
    # x is the position array
    # y is the spike phase array
    if cm and any(x<2) and any(xedges<2):
        x=x*100
        xedges= xedges*100

    H= np.histogram2d(x,y,bins=(xedges,yedges))
    H = gaussian_filter(H[0],sigma=gaussSigma)
    xmesh,ymesh = np.meshgrid(xedges,yedges)
    if plotit:
        ax = plotit
        ax.pcolormesh(xmesh,ymesh,H.T,cmap='jet') # cmap inferno is also good
        ax.set_xlabel('metres')
        ax.set_ylabel('phase')
        ax.set_yticks([0,360,720])
        ax.set_title('Spike Density')

        return H
    else:
        return xmesh,ymesh,H

# plot contour data
def plotContour(z,xedges,yedges,problevel=[0.3],plotit=True):
    X,Y = np.meshgrid(xedges[0:-1],yedges[0:-1])# -1 cz the counts (z) are one less than the edges
    cs1 = plt.contour(X,Y,z,levels=problevel,colors='white')
    if plotit:
        plt.contourf(X,Y,z)
        plt.colorbar()
        plt.clabel(cs1,inline=False)

    return X,Y,cs1

# contour based cluster isolation
def isolateContours(x,y,xedges,yedges,gaussSigma,contourlevels=[0.3],method='maxlen',plotclust=False,plotcon=False,fitEllipse=True):

    xmesh,ymesh,count= plotSpikeDensity(x,y,xedges=xedges,yedges=yedges,gaussSigma=gaussSigma, plotit=False)
    count = count/np.max(count)

    xmesh,ymesh = np.meshgrid(xedges[0:-1],yedges[0:-1]) # -1 cz z is one less
    cs = plt.contour(xmesh,ymesh,count.T,levels=contourlevels,colors='white')

    xc=[]
    yc=[]
    consize=[]
    vertices=[]
    for item in cs.collections:
        for i in item.get_paths():
            v = i.vertices
            vertices.append(v)
            xc.append(v[:, 0])
            yc.append(v[:, 1])
            consize.append(len(v[:,0]))
    contourdata={}
    contourdata['x']=xc
    contourdata['y']=yc
    # levels = cs.levels

    # get the contour with the maximum length
    if method=='maxlen': # include other methods like isolating based on contour area or whether the contour is convex etc.
        ind= np.argmax(consize)
        maxcon_x = contourdata['x'][ind]
        maxcon_y = contourdata['y'][ind]

    # get the points inside the polygon
    x,y = x.flatten(),y.flatten()
    points = np.vstack((x,y)).T
    p = Path(vertices[ind]) # make a polygon for the contour with max length
    grid = p.contains_points(points)

    if plotcon:
        ax=plotcon
        ax.scatter(x,y,c='b',s=2)
        # ax.scatter(x[grid],y[grid],c='k')
        ax.plot(maxcon_x, maxcon_y,lw=2,c='r')

    if fitEllipse: # fits ellipse on the contour with maximum length. this might need more work. Maybe write to fit ellipse on all contours
        # imported here (not at module top) so the scikit-image dependency is only required
        # for callers that actually use isolateContours' ellipse fit -- pip install as
        # "scikit-image" (the PyPI package name); the import name itself stays "skimage".
        from skimage.measure import EllipseModel
        ell = EllipseModel()
        ell.estimate(vertices[ind])
        xcen,ycen,a,b,theta = ell.params
        ellipseparams={}
        ellipseparams['xcentre']=xcen
        ellipseparams['ycentre'] = ycen
        ellipseparams['majorRadius'] = a
        ellipseparams['minorRadius'] = b
        ellipseparams['rotAnticlock'] = theta*180/np.pi

        if plotclust:
            from matplotlib.patches import Ellipse
            ax=plotclust
            ax.scatter(x[grid],y[grid],c='k',s=2)
            ellpatch=Ellipse((xcen,ycen),2*a,2*b,theta*180/np.pi,edgecolor='black') # theta is rotation anticlockwise(given in degrees to the function ellipse)
            ax.add_artist(ellpatch)
            ellpatch.set_clip_box(ax.bbox)
            ellpatch.set_alpha(0.2)
            ellpatch.set_facecolor('red')
#             ax.plot(contourdata['x'][ind], contourdata['y'][ind],lw=2,c='r')
            ax.set_title('xy=' + str(round(xcen,2)) + ',' + str(round(ycen,2))+
                     '\n 2a,2b=' + str(round(2*a,2)) + ',' + str(round(2*b,2)) +
                     ' theta=' + str(round(theta*180/np.pi,2)))
            ax.set_ylabel('degrees')
            ax.set_xlabel('cms')
            ax.set_xlim([min(x)-10, max(x)+10])

    # returns contoour data and the xy data inside the contour with max length provided its a closed polygon
    if fitEllipse:
        return contourdata,consize,x[grid],y[grid], ellipseparams
    else:
        return contourdata,consize,x[grid],y[grid]



# finding the best fit in a 360 degree window
def linRegPrecession(x,y):
    # take the best fit out of 360 fits performed in window of 360 degrees moving by 1 degree
    r2=[]
    intercept=[]
    coef=[]
    step = 1 # degrees
    for i in range(0,360-step,step): #rotating in steps of one degree
        idx= np.where(np.logical_and(y>=i, y<i+360))
        if np.shape(idx)[1]>1:
            x_ = x[idx]
            y_ = y[idx]
            x_ = x_.reshape(-1,1)
            lm= LinearRegression()
            lm.fit(x_,y_)
            if lm.coef_<=0: # take only negative slope values
                r2.append(lm.score(x_,y_)) # check if this is really r2
                intercept.append(lm.intercept_)
                coef.append(lm.coef_)
            else:
                i+=1
        else:
            i +=1
    #print(len(r2))
    # return the phase range with the best linear fit and corresponding scores and intercepts
    bestfit = np.argmax(r2)
    beta = intercept[bestfit] # this is the offset
    slope = coef[bestfit]
    bestr2 = r2[bestfit]
    x_ = x[np.where(np.logical_and(y>bestfit,y<bestfit+360))]
    y_pred = lm.predict(x_.reshape(-1,1))
    phsrng = y[np.where(np.logical_and(y>bestfit, y<=bestfit+360))]

    return slope,bestr2,phsrng,beta,x_,y_pred

#linear fit
def linfit(x,y,plotit=True, deviation=True):
    if not np.any(y) or not np.any(x):
        r2=[]
        slope = []
        offset = []
        y_ = []
        x_ = []
        dev= []
    else:

        from sklearn.linear_model import LinearRegression
        x_ = x.reshape(-1,1)
        reg = LinearRegression().fit(x_,y)
        r2 = reg.score(x_,y)
        slope = reg.coef_
        offset = reg.intercept_
        x_pred = [min(x),max(x)]
        y_ = slope*x_pred + offset

        if plotit:
           ax=plotit
           ax.scatter(x,y,c='black',s=2)
           ax.plot(x_pred,y_,c='red',lw=2)
           ax.set_xlabel('cms')
           ax.set_ylabel('degrees')
           ax.set_title('r2='+ str(np.round(r2,2))+ ' slope='+ str(np.round(slope[0],2)))
           ax.set_ylim([0,720])

        if deviation:
           fittedy = slope*x + offset
           dev = y-fittedy

    op={}
    op['r2'] = r2
    op['slope'] = slope[0]
    op['intercept'] = offset
    op['ypred']=y_
    op['ypred_x'] = x_

    if deviation:
        op['dev'] = dev

    return(op)

# circular linear regression based on Kempter 2012 taken from buzsaki github code circ_regress_cb
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# this is used by buzsaki from the Phillip Berens tool box
def circ_r(alpha,w=None,d=None,dim=0):
    # computes mean resultant vector length for circular data:

#     input:
#     alpha  sample of angles in radians
#     [w     number of incidence in case of binned angle data]
#     [d     spacing of bin centers for binned data, if supplied correction factor is used to correct
#             for bias in estimation of r, in radians]
#     [dim    compute along this dimension, default is 1]

#     if dim argument is specified, all other optional arguments can be left empty

#     PHB 7/6/2008

#     References:
#     Statistical analysis of circular data, N.I. Fisher
#     Topics in circular statistics, S.R. Jammalamadaka et al.
#     Biostatistical Analysis, J. H. Zar

#     Circular Statistics Toolbox for Matlab

#     By Philipp Berens, 2009
#     berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

    if w==None:
        w = np.ones(np.shape(alpha))
    else:
        assert np.shape(w) == np.shape(alpha), 'Input dimensions do not match'

    if d==None:
        d=0 #per default, do not apply correct for binned data

    # compute weighted sum of cos and sin angles
    r = np.sum(w*np.exp(1j*alpha),axis=dim)

    # obtain length
    r = abs(r)/np.sum(w,axis=dim)

    # for data with known spacing, apply correction factor to corre3ct for bias in the
    # estimation of r (See Zar,p 601, equ 261.16)
    if d!=0:
        c = d/2/np.sin(d/2)
        r = c*r

    return r

def costfn(m,x,y):
    # cost function to be minimized
    # parameters: m= line slope, b = line intercept, x= data vector, k = von M concentration ( currently ommitted)
    alpha = y - m*x
    cost = -circ_r(alpha) # mvl should be large for best fit( see kempter fig7) so added -ve so that can min function

    return cost

def kempCorr(theta, t):
    #Correlation described in Kempter et al (2012) see eq3/. See notes above
    #about this not working - might be atypo I've not picked up. Left for
    #completness.
     # t phase values
    #theta - circularised linear position variable
    #First calculate tBar and thetaBar which are just circular mean - use own
    #function for this cb_circMean [which is in CB_Universal folder]. Values
    #cb_circMean works in rads

    # using scipy stats function. confirm that the values are in between 0-2pi
    tBar = sp.stats.circmean(t)
    thetaBar=sp.stats.circmean(theta)

    corr= sum(np.sin(t-tBar)* np.sin(theta-thetaBar)) / np.sqrt( sum(np.sin(t-tBar)**2) * sum(np.sin(theta-thetaBar)**2)) #eq3

    return corr

# perform linear circular regression on the data:

def lcc(x,y,max_slope,con):

    # 1. perform slope optimization using fminbound and find the intercept
    slope,fval,err,numfnc= sp.optimize.fminbound(lambda m: costfn(m,x,y),con[0]*max_slope,con[1]*max_slope,full_output=True)
    S = sum(np.sin(y-slope*x))
    C = sum(np.cos(y-slope*x))
    phase = atan2(S,C)


    regData={}
    regData['slope_opt'] = slope
    regData['fval_opt'] = fval
    regData['phase_opt'] = phase

    # 2. (added by buzsaki lab) now get circular - linear corr coef by converting the circ variables
    # ( eqns 3 and 4 of Kempter 2012). Phi is equivalent to y here ( ie phase angle) and x is x ie position
    # and theta is the linear variable x converted into a circular variable - see Kempter but note that
    # the conversion is always with a +ve gradient so corr will always be +ve

    theta = (abs(slope)*x)%(2*pi) # checked that % which is same as np.remainder is equivalent of matlab mod

    # 3. now that there are two circular variables, take circ-circ regression. Using Kempter method:
    regData['corr'] = kempCorr(theta,x)

    return regData

def circRegress(x,y,k,ppdir,maxslope='default'):
    # check if values are in radians - all values should be between 0 and 2pi,
    # so should have been taken mod(2pi)
    assert np.all(y<(2*np.pi)) and np.all(y>=0), 'Angular values outside the allowed range [0 to 2pi].'

    #Check if there are enough values to get a valid line - n>=3
    if len(x)<3:
        regData={}
        regData['slope_opt']=nan;
        regData['phase_opt']=nan;
        regData['fval_opt']=nan;
        regData['corr']=nan;
        regData['psim']=nan;

        # set constraints on slope min/max for phase precession: if we can be sure that phase precession must
    # be in a certain direction, say because we know that the intrinsic frequency is mostly above theta
    # frequency or vice versa, we can setup the constraints on the slope to exclude spuriously high or
    # low gradients .
    # if ppdir = -1, the slope is constrained to be negative, if ppdir= 1 then constrained to be positive.
    # if empty then the code produces unconstrained optimisation

    # ppdir is an input into the main function
    eps = np.finfo(float).eps # machine epsilon

    h = ppdir
    if h==-1:
        con = [-1, eps]
    elif h==1:
        con = [eps,-1]
    else:
        con = [-1, 1-eps]

    # constrain maximum slope to give at most 360 deg phase precession over the field <<<<<<
    # <<< WILL HAVE TO PLAY WITH THIS FOR CASES WHERE PHASE PRECESSION >720.

    if maxslope=='default':
        max_slope = (4*pi)/(max(x)-min(x))
    elif maxslope=='greater':
        max_slope = (12*pi)/(max(x)-min(x))
    elif maxslope == 'lesser':
        max_slope = (2*pi)/(max(x)-min(x))

    # perform regress
    regData = lcc(x,y,max_slope,con)

    return regData
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def movingavgPhase(x,y,winlen=5,overlap=0.5,method='mean',plotit=True,smSigma=False):
    #confirm no nan in the phs data:
    if any(np.isnan(x)):
        print('there are nans in the data')
    if any(x<2):
        x = x*100 # convert to cms if not
    step = winlen*(1-overlap)
    xbins = np.arange(min(x),max(x),step)
    movingavg=[]
    xxbins=[]
    for j in xbins:
        win = [j,j+winlen]
        idx = np.where((x>win[0])&(x<win[1]))
        if len(y[idx])>5:
            xxbins.append(np.mean(x[idx]))
            if method=='mean':
                movingavg.append(sp.stats.circmean(y[idx],high=360,nan_policy='omit'))
            elif method=='median':
                movingavg.append(np.nanmedian(y[idx]))
        j= win[1]+step

    if plotit:
        plt.scatter(x,y,c='k')
        plt.plot(xxbins,movingavg,c='r',lw=2)
        plt.xlabel('cms')
        plt.ylabel('degrees')
        if smSigma:
            # smav = gaussian_filter1d(movingavg,sigma=smSigma)
            smav = gaussSmooth(movingavg,n=7,sigma=smSigma)
            plt.plot(xxbins,smav,'--')

    return xxbins,movingavg

# different computations to be performed in a moving window. 1. mean 2. median 3. linear fit
def movingwinCompute(x,y,winlen=5,overlap=0.5,method='mean',smSigma=False,plotlin=False):
    minspks = 10
    #confirm no nan in the phs data:
    if any(np.isnan(x)):
        print('there are nans in the data')
    if any(x>1) and any(x<2):
        x = x*100 # convert to cms if in metres
    step = winlen*(1-overlap) # this is in cms
    xbins = np.arange(min(x)-step,max(x),step)
    movingavg=[]
    xxbins=[]
    for j in xbins:
        win = [j,j+winlen]
        idx = np.where((x>win[0])&(x<win[1]))
        if len(y[idx])>minspks:
            xxbins.append(np.mean(x[idx]))
            if method=='mean':
#                 movingavg.append(sp.stats.circmean(y[idx],high=360,nan_policy='omit'))
                movingavg.append(np.nanmean(y[idx]))
            elif method=='median':
                movingavg.append(np.nanmedian(y[idx]))
            elif method=='linfit':
                reg=linfit(x[idx],y[idx],plotit=plotlin)
                movingavg.append(reg['slope'][0])

        j= win[1]+step

    return xxbins,movingavg

def movingDerivative(x,y,winlen=5,overlap=0.5):
    av = np.array(y)
    xx = np.array(x)
    step = winlen*(1-overlap)
    xbins = np.arange(min(xx)-step,max(xx),step)
    der=[]
    xder=[]
    for j in xbins:
        win = [j,j+winlen]
        idx = np.where((xx>win[0])&(xx<win[1]))
        if len(idx)>1:

            dy = av[idx][-1]- av[idx][0] # difference between the first and the last value of the window
            dx = xx[idx][-1]- xx[idx][0]
            der.append(dy/dx)
            xder.append(np.mean(xx[idx]))
    # convert the nans creating by 0/0 divide back to 0
    der = np.nan_to_num(der)
    xder = np.nan_to_num(xder)

    return der,xder

def plotPrecession(x,y,cms=False):
    if cms and any(x<2):
        x = x*100
        plt.xlabel('cms')
    else:
        plt.xlabel('m')
    plt.scatter(x,y) # x and y are for two cycles of theta
    plt.ylim([0,720])
    plt.yticks([0,360,720])
    plt.ylabel('degrees')

def findprefPhaseInh(x,y,outsidefield=[100,150]):
    # x is spikepos, y is spikephase over 2 cycles of theta
    # make sure the units of x and outsidefield are the same.
    if any(x>10) and any(np.array(outsidefield)<2):
        x=x/100
    elif any(x<2) and any(np.array(outsidefield)>10):
        x=x*100

    #indices for a section outside the place field
    ind = np.where((x>outsidefield[0])&(x<outsidefield[1])) # this is in metres
    # indices where the difference is close to 0
    ind0 = np.where((np.diff(y[ind])>0)&(np.diff(y[ind])<0.5)) # increase the diff to 1 degree if not getting enough consistent points
    prefPhase = round(np.nanmedian(y[ind0]),2) # median giving more accurate results than mean
    #precession cycle
    prescycle = [prefPhase, prefPhase+360]

    return prefPhase,prescycle

def GMMcluster(x_,y_,ncopies=2,plotit=True):
    x_,y_ = removeNaNs(x_, y_)
    k= ncopies
    x=[]
    y=[]
    for j in range(ncopies):
        x = np.append(x,x_)
        y = np.append(y,y_+j*360)

    from sklearn.mixture import GaussianMixture

    y = y.reshape(-1,1)
    #define the model
    model= GaussianMixture(n_components=k)
    #fit the model
    model.fit(y)
    #assign a cluster to each example
    labels = model.predict(y)
    # retrieve unique clusters
    clusters = np.unique(labels)

    if plotit:
        for j in range(len(clusters)):
            clusterind = np.where(labels==j)
            plt.scatter(x[clusterind],y[clusterind])

    return clusters,labels,x,y

def findPhaseValley(y,ncopies,bins=50):
    # ensure y is  atleast 2 cycles of theta:
    y_=[]
    for j in range(ncopies):
        y_= np.append(y_,y+j*360)

    h = np.histogram(y_,bins=bins) # fix a bin width here instead
    counts = h[0]
    edges = h[1]
    val = np.argmin(counts)
    phasevalley = edges[val]

    return phasevalley


def calcMehtaSkew(x):
    # this should be calculated on a rate mapns/6518811/interpolate-nan-values-in-a-numpy-a
    # based on 2000 mehta Neuron paper:
        # The skewness was defined in dimensionless units as the ratio of the third moment of the
        # place field firing rate distribution divided by the cube of the standard deviation

    # formula is mentioned in the Spiegel 1994 book cited in mehta 2000.
    # it is the exact same formula as used in scipy.stats.skew
    m3 = np.mean((x- np.mean(x))**3)
    sdcube = (np.std(x))**3
    mehtaskew = m3/sdcube # this gives the same values as skew function

    # scipy.stats.skew : For normally distributed data, the skewness should be about zero.
    # For unimodal continuous distributions, a skewness value greater than zero means that
    # there is more weight in the right tail of the distribution. The function skewtest
    # can be used to determine if the skewness value is close enough to zero, statistically speaking.

    return mehtaskew

def spatialInputstats(a):
    #calculate the first four moments of the skewnorm pdf generated:
    inputmean, inputvar,inputskew, inputkurt = skewnorm.stats(a,moments='mvsk')
    return inputmean, inputvar,inputskew, inputkurt


    # Rajat:
# takes vertices of polygon, all scatter points as inputs
def calcWithinPolygon(coords, xp, yp):
    poly = Polygon(coords)
    xin, yin = [], []
    for x,y in zip(xp, yp):
        p1 = Point(x,y)
        if p1.within(poly):
            xin.append(x)
            yin.append(y)
    return np.array(xin), np.array(yin)


def frRatemap(spkpos,posbins,numpasses,vel):
    # posbins = np.arange(0,2.2,0.05) # default for EI simulations
    spkmap = np.zeros(posbins.shape)
    binsize = posbins[1]-posbins[0]
    occ = (binsize/vel)*numpasses

    for j in range(1,len(posbins)):
        idx = np.where((spkpos>=posbins[j-1])& (spkpos<posbins[j]))
        spkmap[j]= len(idx[0]) # number of spikes in that bin

    occmap = np.zeros(posbins.shape)+ occ # since the number of time each bin is the same and equal to number of multiple passes set
    ratemap = spkmap/occmap

    # gaussratemap = gaussian_filter1d(ratemap,1)
    gaussratemap = gaussSmooth(ratemap,n=4,sigma=1)

    return ratemap, gaussratemap,occmap


# feild detection to remove the few points that are driving the phase range calculation
def fieldDetect(ratemap,posbins,spkpos,spkphs,spkts=None,thresh=0.1):
    # default is set to 10 % fall off of the peak firing rate
    # find points greater than 10 %
    idx = np.where((ratemap>=0.1*max(ratemap)))
    normposbins = (posbins[idx]-min(posbins[idx]))
    normposbins = normposbins/max(normposbins)
    strtfld = posbins[min(idx[0])]
    endfld = posbins[max(idx[0])]

    # to fit two slopes in first and second half of precession
    posofpk = posbins[np.argmax(ratemap)]
    x = spkpos[np.where((spkpos>=strtfld)&(spkpos<=endfld))]
    y= spkphs[np.where((spkpos>=strtfld)&(spkpos<=endfld))]

    #normalize field data:
    x_ = (x-min(x))
    x_=x_/max(x_)

    fielddata={}
    fielddata['spkpos'],fielddata['spkphs'] = removeNaNs(x,y)
    fielddata['normspkpos'],_ = removeNaNs(x_,y)
    fielddata['fldstart'] = strtfld
    fielddata['fldend'] = endfld
    fielddata['fldcentre'] = posofpk
    fielddata['normfldcentre'] = normposbins[np.argmax(ratemap[idx])]
    fielddata['normfldstart'] = min(x_)
    fielddata['normfldend'] = max(x_)

    if spkts:
        t = spkts[np.where((spkpos>=strtfld)&(spkpos<=endfld))]
        _,fielddata['spkts'] = removeNaNs(x,t)

    return fielddata


def calcEIratio(totexc,totinh):

    ei = abs(totexc/totinh) # this is a ratio so is dimensionless
    netexc = totexc+totinh  # this is area under curve so pamp^2

    return ei,netexc


def continuousSpktimes(spktimeperlap,Trun=3):
    'spiketimes and Trun should be in seconds, else convert both to same units'
    # default simulation run time 3 seconds
    spktimecont = []
    for i in range(0,len(spktimeperlap)):
        x = spktimeperlap[i]+ Trun*i
        spktimecont = np.append(spktimecont,x)

    return spktimecont

def TempAutocorr(spktimes):
    'give continous spike times here'
    bins = np.arange(-1000,1020,20)
    count = np.zeros(np.shape(bins))
    for j in range(1,len(spktimes)):
        for k in range(1,j+1):
            if k==j:
                count[50] = count[50]+1
            else:
                isi = (spktimes[k]-spktimes[j])*1000 # converting to ms

                if (isi>=-1000) and (isi<=1000):
                    binindex = int((isi+1000)/20)+1 # 1000 to prevent the sum=0 in case of isi=-1000
                    #chose 1005 to ensure it goes in the right bin
                    count[binindex] = count[binindex]+1
                    count[100-binindex] = count[100-binindex]+1
                else:
                    continue

    # smoothcount = gaussian_filter1d(count,1) # 1 is working best here, check the smoothing calculation
    smoothcount = gaussSmooth(count,n=7,sigma=1)
    isim = (np.diff(spktimes))
    tempcorr={}
    tempcorr['bins'] = bins
    tempcorr['counts'] = smoothcount
    tempcorr['isi'] = isim

    return tempcorr


def precessionFreq (spktimes):
    'give continous spike times here'
    bins = np.arange(-500,503,2) # 2ms bin size for resolution in frequency calculation
    count = np.zeros(np.shape(bins))
    for j in range(1,len(spktimes)):
        for k in range(1,j+1):
            if k==j:
                count[225:275] = 0 # leaving as 0 to remove the centre peak+-50ms
            else:
                isi = (spktimes[k]-spktimes[j])*1000 # converting to ms

                if (isi>=-500) and (isi<=500):
                    binindex = int((isi+500)/2)+1 # 1000 to prevent the sum=0 in case of isi=-1000
                    #chose 1005 to ensure it goes in the right bin
                    count[binindex] = count[binindex]+1
                    count[500-binindex] = count[500-binindex]+1
                else:
                    continue

    # smoothcount = gaussian_filter1d(count,7) # 1 is working best here, check the smoothing calculation
    smoothcount = gaussSmooth(count,n=11,sigma=2)
    isim = (np.diff(spktimes))

    # not ideal for fewer spikes. Resolution issue. Also check with fft. check the matlab function
    idx,_ = find_peaks(smoothcount)
    freq = 1000/np.sort(abs(bins[idx]))[0] # 1/time of the first peak

    return freq

def thetaskipAmp (tempcorr):
    # idx,_ = find_peaks(tempcorr['counts'])
    amp = tempcorr['counts']
    bins = tempcorr['bins']
    # take +-50ms from the 125 and 250 peak
    firstpk = max(amp[np.where((bins>75)&(bins<150))])
    secondpk = max(amp[np.where((bins>200)&(bins<300))])
    skipamp = secondpk/firstpk

    return skipamp

def spinfo(ratemap,occmap): # pass gmap for now
    # spatial info calculation only gaussian smoothed rate map. check adaptive binning for 1d data
    # from Sachins scripts if needed
    infoscore = 0
    if np.nanmax(ratemap)>=0:
        #firing rates in occupied pixels
        rocc = ratemap[occmap>=0]
        occ = occmap[occmap>=0]
        occ = occ/np.nansum(occ)
        # this threshold is used in Jim's program and original Skaggs c program.- you need to avoid pixels with
        #rates = 0, since log(0) is undefined, but using 0.0001 as threshold instead of 0 doesn;t really hurt
        # the info score significantly
        mrate = np.nanmean(rocc)
        for r,p_i in zip(rocc,occ):
            if r > 0.0001:
                infoscore = infoscore+ p_i*(r/mrate)*np.log2(r/mrate)
    else:
        infoscore = 0.0 # setting 0 by Adi instead of nan to avoid str conversion issues during plotting
    if infoscore<0:
        infoscore = 0.0

    return infoscore

def fftPowerRatio(x,sr,frthresh=3,maxnorm=False, plotit=False):
#     sr = 50  # from the xnew - these are number of points after interpolation
    # ts = 1/sr
    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    fftamp = np.abs(X)
    if maxnorm==True:
        fftamp = fftamp/np.max(fftamp) # max normalizing
    freq = n/T
    # th = 1/freq # this is the normalized position ag
    freq = freq[0:N//2]
    fftamp = fftamp[0:N//2]
    lowpowersum = np.sum(fftamp[freq<frthresh-0.1])
    hipowersum = np.sum(fftamp[freq>frthresh-0.1])
    sumratio = np.sum(fftamp[freq<frthresh-0.1])/np.sum(fftamp[freq>frthresh-0.1])
    meanratio = np.mean(fftamp[freq<frthresh-0.1])/np.mean(fftamp[freq>frthresh-0.1])
    ratio1by2 = fftamp[freq==1]/fftamp[freq==2]

    fftdata={}
    fftdata['freq'] = freq
    fftdata['power'] = fftamp
    fftdata['powersumratio'] = sumratio
    fftdata['powermeanratio'] = meanratio
    fftdata['powersumlo'] = lowpowersum
    fftdata['ratio1by2'] = ratio1by2
    fftdata['powersumhi'] = hipowersum

    return fftdata

def objectivequad(x, a, b, c):
    return a * x**2 + b * x + c

def objectivelin(x, s, cee):
    return s * x + cee

def quadfit(x,y):
    popt, _ = curve_fit(objectivequad, x,y)
    # summarize the parameter values
    a, b, c = popt
    # calculate the output for the range
    y_quad = objectivequad(x, a, b, c)
    r2 = r2_score(y,y_quad)

    return r2,a,b,c,y_quad

def lineqnfit(x,y):
    # curve fit
    popt, _ = curve_fit(objectivelin, x, y)
    s,cee = popt # slope and offset
    # calculate the output for the range
    y_line = objectivelin(x, s, cee)
    r2 = r2_score(y,y_line)

    return r2,s,cee,y_line
