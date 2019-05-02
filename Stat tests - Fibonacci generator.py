import numpy as np
import scipy.stats as st
import pylab as py
from datetime import datetime

def fibo_del():
   m = 100
   p = 24  #k
   q = 55 #l

# współczynnik określający zakres generowanych liczb pseudolosowych (od 0 do m-1)
   ciag_wyjsciowy = []
   dl_ciagu = 70
   #initial_values = [8, 6000, 7000, 5000, 3000, 12121, 90000]  # q = len(initial_values)
   initial_values = []
# Loop that auto generates starting values, q = number of initial values
   for v in range(q):
       v = datetime.now().microsecond % m
       initial_values.append(v)
   for n in range(dl_ciagu):
       for i in range(len(initial_values)):

           if i is 0:
               out = (initial_values[p - 1] + initial_values[q - 1]) % m  # the pseudorandom output
           elif 0 < i < len(initial_values) - 1:
               initial_values[i] = initial_values[i + 1]  # shift the array
           else:
               initial_values[i] = out
               ciag_wyjsciowy.append(initial_values[i])
   return ciag_wyjsciowy

# histogram i f-cja gęstości prawdopodob. dla rozkładu normalnego o średniej
# oraz wariancji oszacowanej ze zmiennych
def histogram(x, N_bins):
    n, bins, patches = py.hist(x, N_bins, density=True, facecolor='orange', alpha=0.75)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    y = st.norm.pdf(bincenters, loc=np.mean(x), scale=np.std(x, ddof=1))
    py.plot(bincenters, y, 'r--', linewidth=1)

def porownanie(x):
    fibo_del()
    py.subplot(2, 2, 2)
    histogram(x, 20)
    W, p_sw = st.shapiro(x)
    D, p_ks = st.kstest(x, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
    tytul = 'Test Shapiro-Wilka p: %(sw).2f ' \
            'Test Koł.-Smir. p: %(ks).2f' % {'sw': p_sw, 'ks': p_ks}

    py.title(tytul)
    # statystyki dla pierwszych dziesięciu punktow
    y = x[0:10]
    fibo_del()
    py.subplot(2, 2, 4)
    histogram(y, 20)
    W, p_sw = st.shapiro(y)
    D, p_ks = st.kstest(y, 'norm', args=(np.mean(x), np.std(x, ddof=1)))
    tytul = 'Test Shapiro-Wilka p: %(sw).2f ' \
            'Test Koł.-Smir. p: %(ks).2f' % {'sw': p_sw, 'ks': p_ks}
    py.title(tytul)

#Spr. danych z rozkladem normalnym
x = st.norm.rvs(size=1000, loc=0, scale=10)
py.figure(1)
porownanie(x)
# A teraz zbadajmy dane z rozkładów innych niż normalny:

#Spr. danych z rozkładem T studenta
x = st.t.rvs(df=2, size=1000, loc=0, scale=1)
py.figure(2)
porownanie(x)

#Spr. danych z rozkładem wykładniczym
x = st.expon.rvs(size=1000, loc=0, scale=1)
py.figure(3)
porownanie(x)

py.show()
