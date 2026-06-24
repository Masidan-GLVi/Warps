from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import numpy.ma as ma
import pandas as pd
from astropy.nddata import Cutout2D
from reproject import reproject_interp
from astropy.utils.masked import Masked
import copy

from astropy import modeling
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from astropy.convolution import convolve, Gaussian1DKernel

gauss = Gaussian1DKernel(stddev=1)



'''
ARQUIVOS EXTERNOS UTILIZADOS:
frame-r... - download do quadro (na banda r por ora) que contém o objeto, pelo SDSS
checkrseg.fits - FITS de segmentação obtido pelo sextractor
5791rcat.txt - tabela de catálogo obtida pelo sextractor
'''

#DEFINIÇÕES para eu não precisar copiar e colar TUDO três vezes
def cores(matriz, centro):
	cores = ['b' if (x < centro) else 'r' if (x > centro) else 'g' for x in matriz] #cores dos plots: azul se à esquerda, vermelho se à direita, verde se no meio  (0, 0)
	return cores

def processardados(dados):
	dadosajuste = dados.data.T


	altura = len(dadosajuste)
	centro, centroy = np.unravel_index(dadosajuste.argmax(), dadosajuste.shape)

	altura = len(dadosajuste)
	eixox =  np.arange(altura) - centro

	picos =  []
	picosx = []

	for l in range(len(dadosajuste)):
		fittado = fitter(model, eixox, dadosajuste[l])
		if fittado.amplitude.value > (3.4e-2):
			picos.append(fittado.mean.value)
			picosx.append(l)
	picos = np.array(picos) - fitter(model, eixox, dadosajuste[centro]).mean.value

	alt = [centro for x in picosx]
	picosxcent = [a-b for a, b in zip(picosx, alt)]

	return picosxcent, picos

def graftudo(iteracao, dados):
	fig = plt.figure(layout='constrained', figsize=(8, 8))
	grid = fig.add_gridspec(2, 2)
	deltay = fig.add_subplot(grid[0, 0])
	linha = fig.add_subplot(grid[0, 1])
	zoom = fig.add_subplot(grid[1, :])
	#cria os plots e subplots

	dadosajuste = dados.data.T #transpondo para ler os dados mais facilmente (em linhas), depois a gente retranspõe
	dadosim = dados.data #pega os dados para fazer a imagem

	centro, centroy = np.unravel_index(dadosajuste.argmax(), dadosajuste.shape)

	altura = len(dadosajuste) #como é um quadrado agora, altura = largura
	eixox =  np.arange(altura) - centro #nos dá uma lista de valores [-L/2, +L/2]
#print(eixox)


	picos =  []
	picosx = []

	for l in range(len(dadosajuste)):
		fittado = fitter(model, eixox, dadosajuste[l])#realiza um ajuste gaussiano de cada coluna da imagem original
		fitcenter = fitter(model, eixox, dadosajuste[centro]).mean.value #faz o valor de y do ponto central, para ajustar depois
		if fittado.amplitude.value > (3.4e-2): #impede valores essencialmente nulos (<5 sigma do fundo nesse caso)
			picos.append(fittado.mean.value-fitcenter) #armazena cada pico (amplitude) da gaussiana
			picosx.append(l)
			deltay.plot(eixox+fitcenter, fittado(eixox), color=cores(range(altura), centro)[l], linewidth=0.3) #+eixox+fitcenter para centralizar o pico central no 0

	mapa = copy.copy(plt.get_cmap('inferno'))
	mapa.set_under(color='white')

	linha.imshow(dadosim, cmap=mapa, vmin=0.001, origin='lower')
	linha.scatter(picosx, np.array(picos)+centro-1, s=0.1, marker = ',', c=cores(picosx, centro))
	linha.axhline(altura, color='w', linestyle='--', lw=0.5)
	linha.set_xlabel('x')
	linha.set_ylabel('y')
	#monta o gráfico da imagem bonitinha sobreposta com as médias das gaussianas

	deltay.set_xlim(-30, 30)
	asp = np.diff(deltay.get_xlim())[0] / np.diff(deltay.get_ylim())[0]
	deltay.set_aspect(asp)
	deltay.axvline(0, color='g', linestyle=':')
	deltay.set_xlabel('$\\Delta y$')
	deltay.set_ylabel('Valor do pixel')
	#monta o gráfico das gaussianas

	alt = [centro for x in picosx]
	picosxcent = [a-b for a, b in zip(picosx, alt)]
	maxy = max(abs(np.array(picos)))
	maxx = max(abs(np.array(picosxcent)))

	zoom.scatter(picosxcent, picos, marker = '.', c=cores(picosx, centro))
	zoom.set_ylim(-1.1*maxy, 1.1*maxy)
	zoom.set_xlim(-1.1*maxx, 1.1*maxx)
	zoom.axhline(0, linestyle='--')
	zoom.axvline(0, linestyle='--')
	zoom.set_xlabel('$\\Delta x$')
	zoom.set_ylabel('$\\Delta y$')
	#monta o gráfico ampliado das médias, para melhor visualização

	plt.suptitle(f"{iteracao}ª iteração")
	return picosxcent, picos



	picosxtrim = picosx[picosx.index(0)-espaço+1:picosx.index(0)+espaço] # X centrado no verde
	picostrim = picos[picosx.index(0)-espaço+1:picosx.index(0)+espaço] #realizando um corte para obtermos o eixo central (a linha central onde o desvio vertical é menor que 1 pixel)
	maxx = max(abs(np.array(picosxcent)))
	maxy = max(abs(np.array(picostrim)))



def angfit(picosx, picos):
	#fazemos uma regressão linear com os 50% internos (como no paper do Zee) para obtermos a inclinação "real" do ângulo de posição e rodarmos a galáxia um pouco mais (ou menos) para ficar realmente horizontal.
	espaço = 0
	dely = 0


	while dely < 1:
		espaço+=1
		dely = max(abs(picos[picosx.index(espaço)]), abs(picos[picosx.index(-espaço)]))

	picosxtrim = picosx[picosx.index(0)-espaço+1:picosx.index(0)+espaço]
	picostrim = picos[picosx.index(0)-espaço+1:picosx.index(0)+espaço]


	linhafittada = fitlinear(modelolinear, picosxtrim, picostrim)
	angulofit = np.arctan(linhafittada.slope.value)

	return picosxtrim, picostrim, linhafittada, angulofit



def grafajuste(iteracao, picosx, picos):
	#gráfico do angfit
	fig = plt.figure(figsize = (6, 6))

	picosxtrim, picostrim, linhafittada, angulofit = angfit(picosx, picos)

	maxx = max(abs(np.array(picosxtrim)))
	maxy = max(abs(np.array(picostrim)))

	plt.scatter(picosxtrim, picostrim, marker = '.', c=cores(picosxtrim, 0))
	plt.plot(picosxtrim, linhafittada(picosxtrim), label='Ajuste Linear')
	plt.ylim(-1.1*maxy, 1.1*maxy)
	plt.xlim(-1.1*maxx, 1.1*maxx)
	plt.legend()
	plt.axhline(0, linestyle='--')
	plt.xlabel('$\\Delta x$')
	plt.ylabel('$\\Delta y$')
	plt.title(f'Ajuste da {iteracao}ª iteração')
	plt.text(3, linhafittada.slope.value*3 + linhafittada.intercept.value,  f'$\\Theta = {np.rad2deg(angulofit):.2f} \\degree$', ha = 'left', va='bottom', rotation = np.rad2deg(angulofit), transform_rotates_text=True, rotation_mode='anchor')



def rotecut(angulo, wcss, imagemask, forma, coord, tamanho):
	rot_matrix = np.array([[np.cos(angulo), -np.sin(angulo)], [np.sin(angulo), np.cos(angulo)]]) #matriz de rotação
	wcsrodado = wcss.deepcopy() #para não precisarmos mexer com a original
	pc = wcsrodado.wcs.get_pc() #Returns the PC matrix in read-only form as double array[naxis][naxis].
	wcsrodado.wcs.pc = rot_matrix @ pc

	#Reprojeção e cutout da imagem, também pelo GPT blééé
	data_rot, footprint = reproject_interp((imagemask.filled(0), wcss), wcsrodado, shape_out=forma)
	cutout = Cutout2D(data_rot, position=coord, size=(tamanho*u.arcsec, tamanho*u.arcsec), wcs=wcsrodado)
	return cutout, wcsrodado



def ajustewarp(picosxcent, picos):

	centro = picosxcent.index(0)
	picoszerado = convolve(np.array(picos), gauss)
	#picoszerado = np.array(picos2) - valorcentro
	xesquerda = picosxcent[:centro+1]
	xdireita = picosxcent[centro:]  #com o centro porque polinomial não diverge
	picosesq = np.array(list(picos[:centro+1]))
	picosdir = np.array(list(picos[centro:]))

	plt.figure()
	plt.scatter(picosxcent, picos, marker = '.', c=cores(picosxcent, 0))
	lines=[':','-.', '-']
	warpsesq=[]
	warpsdir=[]

	for grau in range(3):
		#modelo polinomial (1, 2, 3)
		modelo = modeling.polynomial.Polynomial1D(grau+1)
		modelo.c0 = 0 #fixado no 0,0 porque começamos do centro, claro
		modelo.c0.fixed=True
		curvafittadaesq = fitlinear(modelo, xesquerda, picosesq)
		curvafittadadir = fitlinear(modelo, xdireita, picosdir)
		plt.plot(xesquerda, curvafittadaesq(xesquerda), ls=lines[grau], c='g', label=f'Ajuste Linear E grau={grau+1}')
		plt.plot(xdireita, curvafittadadir(xdireita), ls=lines[grau], c='orange', label=f'Ajuste Linear D grau={grau+1}')
		#usando o ponto mais distante de cada lado para calcular o warp
		warpesq = np.arctan(-curvafittadaesq(xesquerda[0])/xesquerda[0])
		warpdir = np.arctan(curvafittadadir(xdireita[-1])/xdireita[-1])
		warpsesq.append(np.rad2deg(warpesq))
		warpsdir.append(np.rad2deg(warpdir))

	plt.legend()
	plt.axhline(0, lw=0.5, c='k', linestyle=':')
	plt.axvline(0, lw=0.5, c='k', linestyle=':')
	plt.xlabel('$\\Delta x$')
	plt.ylabel('$\\Delta y$')
	plt.title(f'Ajuste de curva do warp')

	return warpsesq, warpsdir


# ----------------------------------------------------------------------------------------
	#coordenadas em graus

rag = 159.862000544
degg = 47.947113674



#arquivos
with fits.open('SMmenosceu.fits') as hduloriginal: #abre o arquivo e fecha no fim do processo
	originalD = hduloriginal[0].data
	originalH = hduloriginal[0].header

with fits.open('SMcheckrseg.fits') as hdulseg:
	mask1 = hdulseg[0].data


#coordenadas e ID
wcs = WCS(originalH)
coord = SkyCoord(ra=rag * u.deg, dec=degg * u.deg)
x, y = wcs.world_to_pixel(coord)
x_int = int(np.round(x))
y_int = int(np.round(y))
IDachada = mask1[y_int, x_int]
print("ID no catálogo =", IDachada)
#essa parte foi o GPT que fez, blé


#dados para rotação
dados = pd.read_table("5791rsmoothcat.txt", header=None, delimiter=r"\s+") #lê o arquivo do catálogo
angulorot = dados.at[IDachada-1, 7] #lê a linha (-1 porque é um DataFrame sem cabeçalho, logo não tem linha 0) e coluna (theta) da tabela
angulorotrad = np.deg2rad(angulorot) #conversão para radianos
#print("PA =",angulorot)

mascara=[]
for l in range(len(mask1)):
	masc = np.where(mask1[l]==IDachada, 0, 1)
	mascara.append(masc)
	#checa onde é diferente da galáxia no catálogo, 1 para onde tem, 0 onde não

imagemask = Masked(originalD, mascara)

#tamanho do quadro:
alt = np.shape(imagemask)[0]
larg = np.shape(imagemask)[1]
forma = np.shape(imagemask)



fitter = modeling.fitting.DogBoxLSQFitter()
model = modeling.models.Gaussian1D() #fit gaussiano
fitlinear = modeling.fitting.LinearLSQFitter()
modelolinear = modeling.models.Linear1D() #fit linear para acharmos a inclinação da reta do eixo central



#agora vamos para o ajuste! Yay!


# -------------------------------------------------------------------------------------------------


# LOOP DE ITERAÇÕES

angulofitsoma = 0
angulofit = True
it = 0

while abs(np.rad2deg(angulofit)) > 0.05: #enquanto o meu PA for maior que 0.05 (basicamente garantindo que a galáxia esteja bem na horizonal)

	it+=1 #iterador

	#rotação
	cutout, wcsrodado = rotecut(-angulorotrad-angulofitsoma, wcs, imagemask, forma, coord, 110)

	# hdu_saida = fits.PrimaryHDU(data=cutout.data, header=cutout.wcs.to_header())
	# hdu_saida.writeto(f'5791r-iter{it}.fits', overwrite=True)

	picosx, picos= processardados(cutout)
	angulofit = angfit(picosx, picos)[3]
	angulofitsoma += angulofit
	if it == 10:
		print('Muitas iterações, parando na 10ª!')
		break

print(f'Número de iterações: {it}')

picosxcent, picos = graftudo(it, cutout)
grafajuste(it, picosxcent, picos)

#depois de alinhar a galáxia o máximo possível, faz o cálculo do warp
warpsesq, warpsdir = ajustewarp(picosx, picos)


print(f'Ajuste de grau 1 |, ({warpsesq[0]:.2f}, {warpsdir[0]:.2f})')
print(f'Ajuste de grau 2 |, ({warpsesq[1]:.2f}, {warpsdir[1]:.2f})')
print(f'Ajuste de grau 3 |, ({warpsesq[2]:.2f}, {warpsdir[2]:.2f})')
print(f'O tipo da galáxia é S.' if warpsesq[2] + warpsdir[2] < max(warpsesq[2], warpsdir[2]) else 'O tipo da galáxia é U.')


plt.show()


'''
Fazer o smoothing
achar o PA c/ norte do sextractor.param
Redigir a metodologia
escrever a introdução se der tempo
TER NOVIDADE NO TEXTO PARA 28
2 semanas de junho para escrever o texto
Mandar para a banca até 20 de junho
Defender pelo início de julho

-3.58557089259E-05
 0.000103978028090

'''
