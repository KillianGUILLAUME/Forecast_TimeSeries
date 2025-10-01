import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


###############
# INDEXING
###############

liste=[]
print(liste[i])  #ieme -1 élément de la listee
print(liste[-1])  #dernier élément de la listee


###############
# SLICING
###############

liste[debut:fin:pas]

liste[:3] #début 0 fin 2 pas 1
liste[2:] #début 1 fin dernier pas 1
liste[::-1] #début jusqu'a fin mais à l'envers
liste.insert(i,'') #rajoute '' au ieme élément
liste.extend(liste2) #rajoute une listee à la fin d'une listee

liste.sort(reverse=False) #trie la listee alphabétique ou numérique, reverse=True le fait à l'envers

for index,valeur in enumerate(liste):
    print(index,valeur)
#print l'index puis la valeur

for a,b in zip(liste1,liste2):
    print(a,b)
#print les éléments de la listee 1 et 2 cote à cote, élément par élément avec deux listees différentes (ou peut etre plus)
#s'arrête avec la plus petite listee


###############
# DICTIONNAIRE
###############

dic={
    "key":"value"
}
#sans guillemet pour int
dic.values() #recup toutes les valeurs
dic.keys() #recup toutes les clés
dic.get["nom de la clef cherchée"] #return rien si elle n'existe pas
dic["nv_cle"]="nv_valeur"

dic={ k:v for k,v in enumerate(liste)} #crée dico avec k en clé v en valeur, k indexé par la listeee qui prend en valeur les éléments de la listeee, permet de gagner du temps de calcul
dico2={ prenom:age for prenom,age in zip(liste_age,liste_prenom) }

dic={ k:v for k,v in enumerate(liste) if ...} #permet de mettre des conditions

list(dic.keys()) # crée une liste des clés du dico (meme chose avec .values())


##############
# RANDOM
##############
random.randrange(n) # donne un entier aléatoire entre 0 et n
random.sample(range(n),m)  #génère une liste aléatoire de m nombre, compris entre 0 et n
random.sample(range(n),random.randrange(m)) #génère une liste de nombre alétoire, avec un nombre aléatoire d'élément (entre 0 et m), compris entre 0 et n
random.suffle(liste) #mélange la liste



#############
# NUMPY
#############

np.random.seed(0) #fixe le meme nombre aléatoire

A = np.array([]) # crée une matrice n ligne (nb de crochet) m colonne (nb d'élément dans chaque crochet)
A.shape # permet de voir la shape de la matrice crée
A.size #retourne le nb d'éléments dans le tableau
B = np.zeros((n,m)) #crée matrice de zeros n ligne m colonne 
C = np.random.randn(n,m) # crée matrice n ligne m colonnes rempli avec des nombres aléatoire suivant une N(0,1)

np.linspace(n,m,p) # crée un tableau 1 dim qui répartie équitablement la valeur p entre la position n et m
np.arrange(n,m,p) #crée tableau de la valeur n à m, avec comme pas p

np.hstack((matriceA,matriceB)) #combine horizontalement les deux matrices A et B, A à droite et B à gauche
np.vstack((matriceA,matriceB)) #combine verticalement les deux matrices A et B, A en haut et B en bas

D = np.eye(4) #crée matrice identité
D = D.reshape((n,m)) #avec n*m = D.size de D avant la transformation
D=D.ravel() #applati la matrice en 1 tableau à 1 ligne




data = np.array([1, 2, 3, 4, 5, 6, 7])


# Appliquer une condition avec np.where
new_column = np.where(data > 3, 'High', 'Low') # si j'ai besoin que d'une condition


# Définir les conditions
conditions = [
    data < 3,
    (data >= 3) & (data < 6),
    data >= 6
]

# Définir les résultats correspondants
choices = ['Low', 'Medium', 'High']

# Appliquer np.select
new_column = np.select(conditions, choices) # si j'ai besoin de plusieurs conditions

#utiliser la fonction hstack ensuite

####### TIPS #######

A = np.array([1,2,3])
A = A.reshape((A.shape[O],1))

###########
# SLICING SUR NP
###########
A[i,j] #prend la valeur de A à la ligne i, colonne j
A[:,0] #parcours toutes les lignes de la matrice sur la colonne 0
A[i:j,l:p] #sélectionne le bloc de la ligne i à j, colonne l à p (en pensant que ca commence à l'indice 0,0)
last_5_rows = A[-5:, :]  # -5: sélectionne les 5 dernières lignes, : sélectionne toutes les colonnes
first_5_rows = A[:5, :]  # :5 sélectionne les 5 premières lignes, : sélectionne toutes les colonnes


A.min() # pour trouver min global matrice
A.min(axis=0) #min chaque colonne, axis =1 pour ligne
A.max()
A.argmin(axis=0) #donne position ligne pour min de chaque colonne
A.argsort() #donne combinaison des lignes pour trier correctement tableau sans le modifier

################
# STATISTIC NUMPY
################

A.mean()
A.mean(axis=0)
A.std()
A.var()

np.corrcoef(A) #matrice de variance covariance de A
value, count = np.unique(A, return_counts=True) #compte le nombre de fois qu'un élément apparait dans le tableau
value[count.argsort()] #donne indication comment trier count

np.isnan(A).sum() # compte le nombre de fois qu'il y a des valeurs nan
A[np.isnan(A)]=0

################
# ALGEBRE LINEAIRE NUMPY
################
A.dot(B)  #différent bien sur de B.dot(A)
A.T #transpose A
np.linalg.det(A)
np.linalg.inv(A)
#s'il y a des corrélations linéaires dans une matrice ie la matrice n'est pas inversible
np.linalg.pinv(A)
np.linalg.eig(A) #retourne valeur propre puis vecteur propre

##################
# GRAPHIQUE 3D
##################

f = lambda x,y : np.cos(x) + np.sin(y*np.cos(x))

X = np.linspace(-np.pi,np.pi,100)
Y = np.linspace(-np.pi,np.pi,100)
X,Y=np.meshgrid(X,Y)
Z=f(X,Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,cmap='plasma')




############
# PANDAS
############

df =pd.read_csv('')

df.shape
df.head()
df.drop(['nom colonne 1','nom colonne 2'],axis = 1 )#axis=1 pour enlever les colonnes
df.describe() #donne statistics sur les colonnes etc
df=df.dropna(axis=0) #axis=0 si on veut eliminer des lignes
df['nom colonne'].value_counts() 
df['nom colonne'].value_counts().plot.bar() #créer graphique avec la répartition suivant le dataset
df['nom colonne'].plot()
df.groupby(['nom colonne']).mean() #fait moyenne suivant les différents groupes de cette colonne

df['column']='series'
df['column'][i,j] #slicing le df pour cette colonne entre la ligne i et j-1
df['colum']<k #mask
df[df['column']<j]#bolean indexing, change le df avec ces conditions sur les colonnes
df[['nom 1','nom2']] #créer nouveau df
df.iloc[0:2,0:2] #localisation par index



####################
# PANDAS TIME SERIES
####################

df = pd.read_csv('nom', index_col='Date',parse_date=True)
df['2010':'2015']['nom colonne'].plot()
df.loc['2010':'2015']['nom colonne'].resample('M').plot()
df['2010':'2015']['nom colonne'].resample('M').mean().plot()

df['nom colonne'].resample('W').agg(['mean','std'])
df['2010':'2015']['nom colonne'].rolling(window=7,center=True).mean().plot()
