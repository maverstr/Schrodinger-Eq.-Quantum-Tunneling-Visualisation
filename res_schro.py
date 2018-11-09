# -*- coding: utf-8 -*-
"""
Created on Mon May 15 18:55:46 2017

@author: Verstraeten Maxime; Matricule ULB : 000425582

Ce programme est destiné à afficher la transmission au travers d'une barrière de potentiel constante, pour le cours de Physique Quantique et Statistique.

Le programme se base sur les résultats obtenus dans le cours théorique.
L'entièreté des équations utilisées proviennent du cours et sont référencées autant que possible.
NB : il serait aussi possible de résoudre le système de manière matricielle (à l'aide de numpy pour + de facilités)....
"""

"""""""""""""""""
#################
#####Imports#####
#################
"""""""""""""""""

from matplotlib.widgets import AxesWidget
import cmath as math
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.widgets import Slider, Button
from cmath import exp, phase, sinh, cosh

###Constantes####
from scipy.constants import h, hbar, electron_volt as eV, c, pi

"""""""""""""""""
#################
###Paramètres####
#################
"""""""""""""""""
zoom = 1
precision = 200 #Règle le niveau de discrétisation (le nombre de sous intervalles par unité sur l'axe X )
w = 2 #largeur de la barrière de potentiel (width)
V0 = 0

# points sur l'axe des X
x_max = 5*w
x_min = -x_max
x = np.arange(x_min, x_max, 1/precision) #Vecteur des abscisses

axisColor = "lightgreen"


class VertSlider(AxesWidget):
    """
    Nécessaire pour transformer les sliders originaux de matplotlib (horizontaux uniquement) en sliders verticaux
    Cette classe provient de StackOverflow (http://stackoverflow.com/questions/25934279/add-a-vertical-slider-with-matplotlib)
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None,
                 slidermax=None, dragging=True, **kwargs):
        
        AxesWidget.__init__(self, ax)

        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.valinit = valinit
        self.poly = ax.axhspan(valmin, valinit, 0, 1, **kwargs)

        self.hline = ax.axhline(valinit, 0, 1, color='r', lw=1)

        self.valfmt = valfmt
        ax.set_xticks([])
        ax.set_ylim((valmin, valmax))
        ax.set_yticks([])
        ax.set_navigate(False)

        self.connect_event('button_press_event', self._update)
        self.connect_event('button_release_event', self._update)
        if dragging:
            self.connect_event('motion_notify_event', self._update)
        self.label = ax.text(0.5, 1.03, label, transform=ax.transAxes,
                             verticalalignment='center',
                             horizontalalignment='center')

        self.valtext = ax.text(0.5, -0.03, valfmt % valinit,
                               transform=ax.transAxes,
                               verticalalignment='center',
                               horizontalalignment='center')

        self.cnt = 0
        self.observers = {}

        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax
        self.drag_active = False

    def _update(self, event):
        if self.ignore(event):
            return

        if event.button != 1:
            return

        if event.name == 'button_press_event' and event.inaxes == self.ax:
            self.drag_active = True
            event.canvas.grab_mouse(self.ax)

        if not self.drag_active:
            return

        elif ((event.name == 'button_release_event') or
              (event.name == 'button_press_event' and
               event.inaxes != self.ax)):
            self.drag_active = False
            event.canvas.release_mouse(self.ax)
            return

        val = event.ydata
        if val <= self.valmin:
            if not self.closedmin:
                return
            val = self.valmin
        elif val >= self.valmax:
            if not self.closedmax:
                return
            val = self.valmax

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val

        self.set_val(val)

    def set_val(self, val):
        xy = self.poly.xy
        xy[1] = 0, val
        xy[2] = 1, val
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % val)
        if self.drawon:
            self.ax.figure.canvas.draw_idle()
        self.val = val
        if not self.eventson:
            return
        for cid, func in self.observers.items():
            func(val)

    def on_changed(self, func):
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        try:
            del self.observers[cid]
        except KeyError:
            pass

    def reset(self):
        if (self.val != self.valinit):
            self.set_val(self.valinit)    
            
            

"""""""""""""""""""""""""""
###########################
###gestion des complexes###
###########################
"""""""""""""""""""""""""""

def phaseRadToDeg(nbcomplex):
    phaseDeg = (phase(nbcomplex))*(360/(2*pi))
    return phaseDeg


#Pour afficher correctement les phases sur le plot, on les relie à une couleur. (inspiré et modifié à partir de : http://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array)
from colorsys import hls_to_rgb
def phaseToColor(nbcomplex):
    r = np.abs(nbcomplex)
    arg = np.angle(nbcomplex) 

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    col = np.vectorize(hls_to_rgb)(h,l,s)
    col = np.array(col)
    col = np.transpose(col)
    return (col[0], col[1], col[2])



"""""""""""""""""
#################
###Résolution####
#################
"""""""""""""""""

m = 511e3*eV/(c*c) #Pour la résolution nous considérons la masse d'un électron
lambda_e = 1
E = pow(h,2)/(2*m*pow(lambda_e,2))
k = 2*pi/lambda_e

def resol(V0,w,precision): 
    E = pow(h,2)/(2*m*pow(lambda_e,2))
    
    try: #on veut éviter la division par 0
        1/(V0-E)
    except ValueError:
        E = V0*(1+1e-25) 
        
    K= math.sqrt(2*m*(V0-E))/hbar #équation 5.60
    vcarre = pow(k,2)+pow(K,2) #équation 5.66
    T = exp(-1j*k*w)*(2*k*K)/(2*k*K*cosh(K*w)+1j*(pow(K,2)-pow(k,2))*sinh(K*w)) #5.64

    R = (-1j*T*exp(1j*k*w)*vcarre*sinh(K*w))/(2*k*K) #5.67
    A = 1j*k*(1-R)/K #5.62
    B = 1+R  

    psiArray = [] #Les résultats à plot seront stockés dans ces tableaux
    colorArray = []
    for elem in x : #il suffit d'appliquer le résultat 5.61
        color = (0,0,0)
        if elem < 0: #a gauche de la barrière de pot
            psi = np.cos(k*elem)+np.sin(k*elem)*1j + R*(np.cos(-k*elem)+np.sin(-k*elem)*1j)
            #psi = exp(1j*k*elem) + R*exp(-1j*k*elem)
        if ((elem >= 0) and (elem < w)): #dans la barrière
            psi = A*np.sinh(K*elem) + B*np.cosh(K*elem)
        if (elem >= w): #a droite.  NB : pas d'onde venant de la droite
            psi = T*(np.cos(k*elem)+np.sin(k*elem)*1j)
            #psi = T*exp(1j*k*elem)
        phasePsi = phaseRadToDeg(psi)
        color = phaseToColor(psi)
        module = abs(psi)
        psiArray.append(module)
        colorArray.append(color)
        
        print("phases :",phasePsi)
        print("modules :",module)
    print("T :",T)
    print("K :",K)
    return (psiArray,colorArray)


def drawBarriere(V0,w,precision):
     #on crée un vecteur contenant (len(x)/2) fois le point de coord y = 0 pour dessiner une ligne horizontale de longueur len(x)/2
        #on prend len(x)/2 car la barrière est placée en x = 0 et il y a autant de points à gauche qu'à droite du 0
    barrier = np.where((x<0) | (x>w), 0, 2*V0/E)
    return barrier

"""""""""""""""
###############
###Affichage### 
###############
"""""""""""""""   
def plotGraph(V0,w,zoom,precision):
    ax.clear()
    ax2.clear()
    (psiArray,colorArray) = resol(V0,w,precision) #On récupère les valeurs récupérées par résolution de l'équation
    ax.plot(x, psiArray, linewidth=0.5, color='black') #on dessine la fonction psi
    ax.plot(x, drawBarriere(V0,w,precision), linewidth=1, color='black') #on dessine la barrière
    ax.set_xlabel("position en X")
    ax.set_ylabel("$|\psi(x)|$")
    ax.set_xlim([-5*(1/zoom), 5*(1/zoom)])
    ax.set_ylim([-0.5,4])
    ax2.set_ylim([-0.8, 2])
    scale_factor = int(round(1000*(1/precision))) # pour accélérer l'affichage (le plus gourmand en temps d'exécution)
    for p in range(len(colorArray)//scale_factor): #on colorie la fonction en fct des phases
        ax.fill_between(x[p*scale_factor:(p+1)*scale_factor+1],psiArray[p*scale_factor:(p+1)*scale_factor+1],color=colorArray[p*scale_factor])





### la partie concernant la gestion des sliders partie est fortement inspirée de la doc sur les widgets de matplotlib (https://matplotlib.org/api/widgets_api.html)
###Ainsi que certains codes provenant de StackOverflow (http://stackoverflow.com/questions/6697259/interactive-matplotlib-plot-with-two-sliders)

#les "event handlers" : bouton reset pressé et slider bougés 
redrawOnChange = True

def reset_button_on_clicked(mouse_event):
    global redrawOnChange
    redrawOnChange=False
    w_slider.reset()
    V_slider.reset()
    zoom_slider.reset()
    prec_slider.reset()
    V = V_slider.val*E
    plotGraph(V,w,zoom,precision)
    fig.canvas.draw_idle()
    redrawOnChange=True
    
def sliders_on_changed(val):
    V = V_slider.val*E
    w = w_slider.val
    zoom = zoom_slider.val
    precision = int(round(prec_slider.val)) # /!!!\ il faut bien arrondir les valeurs pour avoir un nombre entier (et positif)
    if redrawOnChange:
        plotGraph(V,w,zoom,precision)
        fig.canvas.draw_idle()
    
# creation du figure plot
fig = plot.figure(figsize=(8,5))
plot.rcParams.update({'font.size': 10})



##maximiser la fenêtre
#fig_manager = plot.get_current_fig_manager()
#if hasattr(fig_manager, 'window'):
#    fig_manager.window.showMaximized()
    

#creation de l' "Axes" et ajustement de sa position
ax = fig.add_subplot(111)
ax2 = ax.twinx()
fig.subplots_adjust(left=0.1,right=0.6, bottom=0.25)

    
# création des Sliders
V_slider_ax  = fig.add_axes([0.7, 0.25, 0.03, 0.65])
V_slider = VertSlider(V_slider_ax, "V0/E", -0.8, 2, valinit=0)
w_slider_ax = fig.add_axes([0.1, 0.1, 0.5, 0.03])
w_slider = Slider(w_slider_ax, 'width', 0.1, 5.0, valinit=w)
zoom_slider_ax = fig.add_axes([0.1, 0.05, 0.5, 0.03])
zoom_slider = Slider(zoom_slider_ax, 'zoom', 0.1, 2.0, valinit=zoom)
prec_slider_ax  = fig.add_axes([0.8, 0.25, 0.03, 0.65])
prec_slider = VertSlider(prec_slider_ax, "precision", 1, 500, valinit=precision)

#configuration des events handlers
V_slider.on_changed(sliders_on_changed)
w_slider.on_changed(sliders_on_changed)
zoom_slider.on_changed(sliders_on_changed)
prec_slider.on_changed(sliders_on_changed)


# création du Bouton reset
reset_button_ax = fig.add_axes([0.7, 0.1, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color = axisColor, hovercolor='0.975')

#configuration des events handlers
reset_button.on_clicked(reset_button_on_clicked)

#calcul et affichage des graphiques
plotGraph(V0,w,zoom,precision)

#creation du titre
plot.title("Transmission et Réflexion d'une Particule au travers d'un potentiel cst", y = 21,x = -3)
# mise à jour de la fenêtre
plot.show()