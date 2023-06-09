# Alan García Justel
# Sistemas de apoyo a la decisión
# 3/3/2023
# Script encargado de clasificar nuevas instancias cargando
# un modelo previamente entrenado
#

from getopt import getopt
from sys import exit, argv, version_info
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Variables globales
OUTPUT_PATH     = ""                            # Path de los archivos de salida
INPUT_TEST      = "SantanderTraHalfHalf.csv"    # Path del archivo de datos a clasificar
INPUT_MODEL     = "./models/model.sav"          # Path del modelo
TARGET_NAME     = ""                            # Nombre de la columna que contiene la clase real
PREPROCESADO    = True                          # ¿Preprocesamos los datos de entrada? -> Cat2Num, Missing values y Escalado

#######################################################################################
#                              ARGUMENTS AND OPTIONS                                  #
#######################################################################################
def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Usage: entrenar.py <optional-args>")
    print("The options supported by entrenar are:")
    print(f"-h, --help          show the usage")
    print(f"-o, --output        output file path for the weights                DEFAULT: ./{OUTPUT_PATH}")
    print(f"-i, --input         input file path of the data                     DEFAULT: ./{INPUT_TEST}")
    print(f"-m, --model         input file path of the model                    DEFAULT: ./{INPUT_MODEL}")
    print(f"--no-preprocesing   disables preprocesing of the input data         DEFAULT: Preprocesing: {PREPROCESADO}")
    print(f"-t --target         target name to predict                          DEFAULT: {TARGET_NAME}")
    print("")
    
    print(f"Example: clasificarItemsNuevos.py -i iris.csv -m ./models/KNN-k:3-p:1-w:uniform.sav -t Especie")

    # Salimos del programa
    exit(1)

def load_options(options):
    # PRE: argumentos especificados por el usuario
    # POST: registramos la configuración del usuario en las variables globales
    global INPUT_TEST, INPUT_MODEL, OUTPUT_PATH, PREPROCESADO, TARGET_NAME

    for opt,arg in options:
        if opt in ('-h', '--help'):
            usage()
        elif opt in ('-o', '--output'):
            OUTPUT_PATH = str(arg)
        elif opt in ('-i', '--input'):
            INPUT_TEST = str(arg)
        elif opt in ('-m', '--model'):
            INPUT_MODEL = str(arg)
        elif opt == '--no-preprocesing':
            PREPROCESADO = False
        elif opt in ('-t', '--target'):
            TARGET_NAME = str(arg)

def show_script_options():
    # PRE: ---
    # POST: imprimimos las configuración del script
    print("entrenar.py configuration:")
    print(f"-o                  output file path            -> {OUTPUT_PATH}")
    print(f"-i                  input test file path        -> {INPUT_TEST}")
    print(f"-m                  input model file path       -> {INPUT_MODEL}")
    print(f"--no-preprocesing   preprocesing data           -> {PREPROCESADO}")
    print(f"-t                  target name                 -> {TARGET_NAME}")

    print("")

#######################################################################################
#                               METHODS AND FUNCTIONS                                 #
#######################################################################################
def coerce_to_unicode(x):
    if version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)
    
    # Si no es anterior a la version 3 de python
    return str(x)

def atributos_excepto(atributos, excepciones):
    # PRE: lista completa de atributos y lista de aquellos que no queremos seleccionar
    # POST: devolvemos una lista de atributos
    atribs = []

    for a in atributos:
        if a not in excepciones:
            atribs.append(a)

    return atribs

def estandarizar_tipos_de_datos(dataset, categorical_features, numerical_features, text_features):
    # PRE: dataset y listas qué atributos son categóricos, numéricos y de texto del dataset
    # POST: devuelve las features categoriales y de texto en formato unicode y las numéricas en formato double o epoch (si son fechas)
    for feature in categorical_features:
        dataset[feature] = dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        dataset[feature] = dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(dataset[feature].dtype, 'base') and dataset[feature].dtype.base == np.dtype('M8[ns]')):
            dataset[feature] = datetime_to_epoch(dataset[feature])
        else:
            dataset[feature] = dataset[feature].astype('double')

def obtener_lista_impute_para(atributos, impute_with, excepciones):
    # PRE: lista de atributos y string indicando con qué valor los imputamos
    # POST: lista del estilo: [{"feature": atrib[i], "impute_with": impute_with}]
    lista = []
    for a in atributos:
        if a not in excepciones:
            entrada = {"feature" : a, "impute_with": impute_with}
            lista.append(entrada)

    return lista

def obtener_lista_rescalado_para(atributos, rescale_with, excepciones):
    # PRE: lista de atributos y string indicando con qué valor reescalamos
    # POST: diccionario del estilo: {'num_var45_ult1': 'AVGSTD', ... }

    diccionario = {}
    for a in atributos:
        if a not in excepciones:
            diccionario[a] = rescale_with;

    return diccionario

def preprocesar_datos(dataset, drop_rows_when_missing, impute_when_missing, rescale_features):
    # PRE: Conjunto completo de datos para ajustar nuestro algoritmo
    # POST: Devuelve el dataset preprocesado

    # Borramos las instancias que tengan missing values
    for feature in drop_rows_when_missing:
        dataset = dataset[dataset[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Inputamos los datos faltantes en función del método elegido en la variable inpute_when_missing
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v1 = dataset[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v1 = dataset[feature['feature']].median()
            
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v1 = 'NULL_CATEGORY'
            
        elif feature['impute_with'] == 'MODE':
            v1 = dataset[feature['feature']].value_counts().index[0]
            
        elif feature['impute_with'] == 'CONSTANT':
            v1 = feature['value']
            
        dataset[feature['feature']] = dataset[feature['feature']].fillna(v1)
        
        
        s1 = f"- Test feature {feature['feature']} with value {str(v1)}"
        print("Imputed missing values\t%s" % (s1))

#######################################################################################
#                                    MAIN PROGRAM                                     #
#######################################################################################
def main():
    # Entrada principal del programa
    print("---- Iniciando main...")
    print(f"---- Cargando el modelo {INPUT_MODEL}")
    clf = pickle.load(open(INPUT_MODEL, 'rb'))

    print(f"---- Cargando los datos a clasificar {INPUT_TEST}")
    ml_dataset = pd.read_csv(INPUT_TEST)
    print(ml_dataset.head(5))

    # Seleccionamos atributos son relevantes para la clasificación
    atributos = ml_dataset.columns # Todos los atributos del dataset
    ml_dataset = ml_dataset[atributos]

    print("---- Estandarizamos en Unicode y pasamos de atributos categoricos a numericos")
    categorical_features = []
    text_features = []
    numerical_features = atributos_excepto(ml_dataset.columns, [TARGET_NAME] + categorical_features + text_features)
    
    # Ponemos los datos en un formato común
    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)
    
    if PREPROCESADO:
        print("---- Preprocesamos los datos.")

        # Convertir variables categoricas en numericas
        ml_dataset[categorical_features] = ml_dataset[categorical_features].apply(lambda x: pd.factorize(x)[0])

        # Definimos los parámetros del preprocesado
        # TODO: Establecer imputacion con media a todos los atributos por defecto y reescalado con AVGSTD
        drop_ft = [] # ['num_var45_ult1', ...] array
        # [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'}, ...] array de diccionarios
        #imput_ft = [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var45_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_reemb_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_compra_var44_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var37_cte', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var40', 'impute_with': 'MEAN'}, {'feature': 'num_var12_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var44', 'impute_with': 'MEAN'}, {'feature': 'ind_var8', 'impute_with': 'MEAN'}, {'feature': 'ind_var24_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var5', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var1', 'impute_with': 'MEAN'}, {'feature': 'ind_var8_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var13_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var33_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var12_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var37_med_ult2', 'impute_with': 'MEAN'}, {'feature': 'num_var7_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_hace2', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_hace2', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_ent_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_venta_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_corto', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var8_0', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var13_largo_0', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var45_hace3', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var13_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_var43_emit_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var45_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var13_corto_0', 'impute_with': 'MEAN'}, {'feature': 'num_var8', 'impute_with': 'MEAN'}, {'feature': 'num_var4', 'impute_with': 'MEAN'}, {'feature': 'num_var5', 'impute_with': 'MEAN'}, {'feature': 'num_var1', 'impute_with': 'MEAN'}, {'feature': 'ind_var12_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var33_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var9_cte_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var39_vig_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var14_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var10_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var37_0', 'impute_with': 'MEAN'}, {'feature': 'num_var13_largo', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_aport_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var26_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var40_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var41_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var14', 'impute_with': 'MEAN'}, {'feature': 'ind_var12', 'impute_with': 'MEAN'}, {'feature': 'ind_var13', 'impute_with': 'MEAN'}, {'feature': 'ind_var19', 'impute_with': 'MEAN'}, {'feature': 'ind_var26_cte', 'impute_with': 'MEAN'}, {'feature': 'ind_var17', 'impute_with': 'MEAN'}, {'feature': 'ind_var1_0', 'impute_with': 'MEAN'}, {'feature': 'num_var25_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var43_emit_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var22_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var22_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var13', 'impute_with': 'MEAN'}, {'feature': 'saldo_var12', 'impute_with': 'MEAN'}, {'feature': 'num_var6_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var14', 'impute_with': 'MEAN'}, {'feature': 'saldo_var17', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var32_cte', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var30_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var25', 'impute_with': 'MEAN'}, {'feature': 'ind_var26', 'impute_with': 'MEAN'}, {'feature': 'imp_trans_var37_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var33_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var24', 'impute_with': 'MEAN'}, {'feature': 'imp_var7_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_ent_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var17_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_med_var45_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var13_0', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var20', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var20', 'impute_with': 'MEAN'}, {'feature': 'ind_var25_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var24', 'impute_with': 'MEAN'}, {'feature': 'saldo_var26', 'impute_with': 'MEAN'}, {'feature': 'saldo_var25', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var40_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var37', 'impute_with': 'MEAN'}, {'feature': 'ind_var39', 'impute_with': 'MEAN'}, {'feature': 'ind_var25_cte', 'impute_with': 'MEAN'}, {'feature': 'num_var24_0', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_compra_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var32', 'impute_with': 'MEAN'}, {'feature': 'num_reemb_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var33', 'impute_with': 'MEAN'}, {'feature': 'num_venta_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var30', 'impute_with': 'MEAN'}, {'feature': 'saldo_var31', 'impute_with': 'MEAN'}, {'feature': 'ind_var31', 'impute_with': 'MEAN'}, {'feature': 'saldo_var30', 'impute_with': 'MEAN'}, {'feature': 'saldo_var33', 'impute_with': 'MEAN'}, {'feature': 'saldo_var32', 'impute_with': 'MEAN'}, {'feature': 'ind_var14_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var5_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var37', 'impute_with': 'MEAN'}, {'feature': 'ind_var37_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_largo', 'impute_with': 'MEAN'}, {'feature': 'saldo_var13_corto', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var44_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var39_0', 'impute_with': 'MEAN'}, {'feature': 'num_var43_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'var21', 'impute_with': 'MEAN'}, {'feature': 'saldo_var40', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_var42', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var17_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var44', 'impute_with': 'MEAN'}, {'feature': 'num_var42_0', 'impute_with': 'MEAN'}, {'feature': 'delta_num_reemb_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var20_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_reemb_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_compra_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var29_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var9_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_num_reemb_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'var15', 'impute_with': 'MEAN'}, {'feature': 'imp_compra_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var30_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var5', 'impute_with': 'MEAN'}, {'feature': 'saldo_var8', 'impute_with': 'MEAN'}, {'feature': 'delta_num_aport_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var1', 'impute_with': 'MEAN'}, {'feature': 'ind_var17_0', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var33_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_hace2', 'impute_with': 'MEAN'}, {'feature': 'ind_var32_0', 'impute_with': 'MEAN'}, {'feature': 'imp_venta_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_hace2', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_hace3', 'impute_with': 'MEAN'}, {'feature': 'delta_num_compra_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var7_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_hace2', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_num_aport_var33_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_var41_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var13_corto_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_hace2', 'impute_with': 'MEAN'}, {'feature': 'delta_num_venta_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'var38', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var5_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var8_ult3', 'impute_with': 'MEAN'}, {'feature': 'var36', 'impute_with': 'MEAN'}, {'feature': 'num_sal_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var26_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var39_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var17_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var10cte_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var31_0', 'impute_with': 'MEAN'}, {'feature': 'num_var22_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var22_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var20', 'impute_with': 'MEAN'}, {'feature': 'imp_compra_var44_hace3', 'impute_with': 'MEAN'}, {'feature': 'imp_sal_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var25', 'impute_with': 'MEAN'}, {'feature': 'num_var24', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var26', 'impute_with': 'MEAN'}, {'feature': 'num_var44_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var6_0', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_aport_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var13_largo_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_reemb_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var5_0', 'impute_with': 'MEAN'}, {'feature': 'num_var29_0', 'impute_with': 'MEAN'}, {'feature': 'num_var12', 'impute_with': 'MEAN'}, {'feature': 'num_var14', 'impute_with': 'MEAN'}, {'feature': 'num_var13', 'impute_with': 'MEAN'}, {'feature': 'num_var32_0', 'impute_with': 'MEAN'}, {'feature': 'num_var17', 'impute_with': 'MEAN'}, {'feature': 'ind_var43_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_trasp_var11_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_corto_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_reemb_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_var40', 'impute_with': 'MEAN'}, {'feature': 'num_var42', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var33_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var17_0', 'impute_with': 'MEAN'}, {'feature': 'num_var44', 'impute_with': 'MEAN'}, {'feature': 'ind_var44_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var29_0', 'impute_with': 'MEAN'}, {'feature': 'num_var20_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var13_largo', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var33_hace3', 'impute_with': 'MEAN'}, {'feature': 'var3', 'impute_with': 'MEAN'}, {'feature': 'num_med_var22_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var13_corto', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_largo_0', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_aport_var33_1y3', 'impute_with': 'MEAN'}, {'feature': 'delta_num_aport_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var17_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var30', 'impute_with': 'MEAN'}, {'feature': 'num_var32', 'impute_with': 'MEAN'}, {'feature': 'num_var31', 'impute_with': 'MEAN'}, {'feature': 'num_var33', 'impute_with': 'MEAN'}, {'feature': 'num_var31_0', 'impute_with': 'MEAN'}, {'feature': 'num_var35', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var17_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var33_0', 'impute_with': 'MEAN'}, {'feature': 'num_var37', 'impute_with': 'MEAN'}, {'feature': 'num_var39', 'impute_with': 'MEAN'}, {'feature': 'num_var1_0', 'impute_with': 'MEAN'}, {'feature': 'imp_var43_emit_ult1', 'impute_with': 'MEAN'}] 
        imput_ft = obtener_lista_impute_para(ml_dataset.columns, "MEAN", [TARGET_NAME])
        # {'num_var45_ult1': 'AVGSTD', ... } diccionario
        #res_ft = {'num_var45_ult1': 'AVGSTD', 'num_op_var39_ult1': 'AVGSTD', 'num_op_var40_comer_ult3': 'AVGSTD', 'num_var45_ult3': 'AVGSTD', 'num_aport_var17_ult1': 'AVGSTD', 'delta_imp_reemb_var17_1y3': 'AVGSTD', 'num_compra_var44_hace3': 'AVGSTD', 'ind_var37_cte': 'AVGSTD', 'num_op_var39_ult3': 'AVGSTD', 'ind_var40': 'AVGSTD', 'num_var12_0': 'AVGSTD', 'num_op_var40_comer_ult1': 'AVGSTD', 'ind_var44': 'AVGSTD', 'ind_var8': 'AVGSTD', 'ind_var24_0': 'AVGSTD', 'ind_var5': 'AVGSTD', 'num_op_var41_hace3': 'AVGSTD', 'ind_var1': 'AVGSTD', 'ind_var8_0': 'AVGSTD', 'num_op_var41_efect_ult3': 'AVGSTD', 'num_op_var41_hace2': 'AVGSTD', 'num_op_var39_hace3': 'AVGSTD', 'num_op_var39_hace2': 'AVGSTD', 'num_aport_var13_hace3': 'AVGSTD', 'num_aport_var33_hace3': 'AVGSTD', 'num_meses_var12_ult3': 'AVGSTD', 'num_op_var41_efect_ult1': 'AVGSTD', 'num_var37_med_ult2': 'AVGSTD', 'num_var7_recib_ult1': 'AVGSTD', 'saldo_medio_var33_hace2': 'AVGSTD', 'saldo_medio_var33_hace3': 'AVGSTD', 'saldo_medio_var8_hace3': 'AVGSTD', 'saldo_medio_var8_hace2': 'AVGSTD', 'imp_op_var39_ult1': 'AVGSTD', 'num_ent_var16_ult1': 'AVGSTD', 'delta_imp_venta_var44_1y3': 'AVGSTD', 'imp_op_var39_efect_ult1': 'AVGSTD', 'ind_var13_0': 'AVGSTD', 'ind_var13_corto': 'AVGSTD', 'saldo_medio_var5_ult3': 'AVGSTD', 'imp_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var5_ult1': 'AVGSTD', 'num_op_var40_efect_ult1': 'AVGSTD', 'num_var8_0': 'AVGSTD', 'imp_op_var39_comer_ult1': 'AVGSTD', 'num_var13_largo_0': 'AVGSTD', 'imp_op_var39_comer_ult3': 'AVGSTD', 'num_var45_hace3': 'AVGSTD', 'imp_aport_var13_hace3': 'AVGSTD', 'num_var43_emit_ult1': 'AVGSTD', 'num_var45_hace2': 'AVGSTD', 'num_var13_corto_0': 'AVGSTD', 'num_var8': 'AVGSTD', 'num_var4': 'AVGSTD', 'num_var5': 'AVGSTD', 'num_var1': 'AVGSTD', 'ind_var12_0': 'AVGSTD', 'num_op_var40_hace2': 'AVGSTD', 'num_var33_0': 'AVGSTD', 'ind_var9_cte_ult1': 'AVGSTD', 'imp_op_var40_ult1': 'AVGSTD', 'num_meses_var39_vig_ult3': 'AVGSTD', 'num_var14_0': 'AVGSTD', 'ind_var10_ult1': 'AVGSTD', 'num_var37_0': 'AVGSTD', 'num_var13_largo': 'AVGSTD', 'delta_imp_aport_var13_1y3': 'AVGSTD', 'saldo_medio_var12_hace3': 'AVGSTD', 'ind_var26_0': 'AVGSTD', 'saldo_medio_var12_hace2': 'AVGSTD', 'num_var40_0': 'AVGSTD', 'ind_var41_0': 'AVGSTD', 'ind_var14': 'AVGSTD', 'ind_var12': 'AVGSTD', 'ind_var13': 'AVGSTD', 'ind_var19': 'AVGSTD', 'ind_var26_cte': 'AVGSTD', 'ind_var17': 'AVGSTD', 'ind_var1_0': 'AVGSTD', 'num_var25_0': 'AVGSTD', 'ind_var43_emit_ult1': 'AVGSTD', 'num_var22_hace2': 'AVGSTD', 'num_var22_hace3': 'AVGSTD', 'saldo_var13': 'AVGSTD', 'saldo_var12': 'AVGSTD', 'num_var6_0': 'AVGSTD', 'saldo_var14': 'AVGSTD', 'saldo_var17': 'AVGSTD', 'imp_op_var41_efect_ult3': 'AVGSTD', 'ind_var32_cte': 'AVGSTD', 'imp_op_var41_efect_ult1': 'AVGSTD', 'ind_var30_0': 'AVGSTD', 'ind_var25': 'AVGSTD', 'ind_var26': 'AVGSTD', 'imp_trans_var37_ult1': 'AVGSTD', 'num_meses_var33_ult3': 'AVGSTD', 'ind_var24': 'AVGSTD', 'imp_var7_recib_ult1': 'AVGSTD', 'imp_ent_var16_ult1': 'AVGSTD', 'imp_aport_var17_hace3': 'AVGSTD', 'num_med_var45_ult3': 'AVGSTD', 'num_var13_0': 'AVGSTD', 'imp_op_var41_comer_ult1': 'AVGSTD', 'imp_op_var41_comer_ult3': 'AVGSTD', 'saldo_var20': 'AVGSTD', 'imp_aport_var17_ult1': 'AVGSTD', 'ind_var20': 'AVGSTD', 'ind_var25_0': 'AVGSTD', 'saldo_var24': 'AVGSTD', 'saldo_var26': 'AVGSTD', 'saldo_var25': 'AVGSTD', 'num_op_var41_comer_ult3': 'AVGSTD', 'num_op_var41_comer_ult1': 'AVGSTD', 'ind_var40_0': 'AVGSTD', 'ind_var37': 'AVGSTD', 'ind_var39': 'AVGSTD', 'ind_var25_cte': 'AVGSTD', 'num_var24_0': 'AVGSTD', 'delta_imp_compra_var44_1y3': 'AVGSTD', 'num_aport_var13_ult1': 'AVGSTD', 'ind_var32': 'AVGSTD', 'num_reemb_var13_ult1': 'AVGSTD', 'saldo_medio_var33_ult3': 'AVGSTD', 'ind_var33': 'AVGSTD', 'num_venta_var44_ult1': 'AVGSTD', 'ind_var30': 'AVGSTD', 'saldo_var31': 'AVGSTD', 'ind_var31': 'AVGSTD', 'saldo_var30': 'AVGSTD', 'saldo_var33': 'AVGSTD', 'saldo_var32': 'AVGSTD', 'ind_var14_0': 'AVGSTD', 'saldo_medio_var33_ult1': 'AVGSTD', 'num_var5_0': 'AVGSTD', 'saldo_var37': 'AVGSTD', 'ind_var37_0': 'AVGSTD', 'ind_var13_largo': 'AVGSTD', 'saldo_var13_corto': 'AVGSTD', 'num_meses_var44_ult3': 'AVGSTD', 'num_var39_0': 'AVGSTD', 'num_var43_recib_ult1': 'AVGSTD', 'var21': 'AVGSTD', 'saldo_var40': 'AVGSTD', 'saldo_medio_var17_ult1': 'AVGSTD', 'saldo_var42': 'AVGSTD', 'saldo_medio_var17_ult3': 'AVGSTD', 'saldo_var44': 'AVGSTD', 'num_var42_0': 'AVGSTD', 'delta_num_reemb_var13_1y3': 'AVGSTD', 'saldo_medio_var13_largo_ult1': 'AVGSTD', 'num_op_var39_comer_ult3': 'AVGSTD', 'num_op_var39_comer_ult1': 'AVGSTD', 'ind_var20_0': 'AVGSTD', 'num_op_var41_ult1': 'AVGSTD', 'num_reemb_var17_ult1': 'AVGSTD', 'saldo_medio_var13_largo_ult3': 'AVGSTD', 'num_compra_var44_ult1': 'AVGSTD', 'num_op_var41_ult3': 'AVGSTD', 'num_meses_var29_ult3': 'AVGSTD', 'imp_op_var41_ult1': 'AVGSTD', 'ind_var9_ult1': 'AVGSTD', 'delta_num_reemb_var17_1y3': 'AVGSTD', 'var15': 'AVGSTD', 'imp_compra_var44_ult1': 'AVGSTD', 'imp_op_var40_efect_ult3': 'AVGSTD', 'imp_op_var40_efect_ult1': 'AVGSTD', 'num_var30_0': 'AVGSTD', 'saldo_var5': 'AVGSTD', 'saldo_var8': 'AVGSTD', 'delta_num_aport_var17_1y3': 'AVGSTD', 'saldo_medio_var8_ult3': 'AVGSTD', 'saldo_var1': 'AVGSTD', 'ind_var17_0': 'AVGSTD', 'num_aport_var33_ult1': 'AVGSTD', 'saldo_medio_var13_corto_hace2': 'AVGSTD', 'ind_var32_0': 'AVGSTD', 'imp_venta_var44_ult1': 'AVGSTD', 'saldo_medio_var5_hace2': 'AVGSTD', 'saldo_medio_var13_corto_hace3': 'AVGSTD', 'saldo_medio_var5_hace3': 'AVGSTD', 'delta_num_compra_var44_1y3': 'AVGSTD', 'saldo_medio_var44_hace3': 'AVGSTD', 'ind_var7_recib_ult1': 'AVGSTD', 'saldo_medio_var44_hace2': 'AVGSTD', 'saldo_medio_var8_ult1': 'AVGSTD', 'delta_num_aport_var33_1y3': 'AVGSTD', 'num_var41_0': 'AVGSTD', 'num_op_var39_efect_ult1': 'AVGSTD', 'num_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace3': 'AVGSTD', 'num_meses_var13_corto_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace2': 'AVGSTD', 'delta_num_venta_var44_1y3': 'AVGSTD', 'var38': 'AVGSTD', 'num_meses_var5_ult3': 'AVGSTD', 'num_meses_var8_ult3': 'AVGSTD', 'var36': 'AVGSTD', 'num_sal_var16_ult1': 'AVGSTD', 'num_var26_0': 'AVGSTD', 'saldo_medio_var44_ult3': 'AVGSTD', 'ind_var39_0': 'AVGSTD', 'saldo_medio_var44_ult1': 'AVGSTD', 'num_aport_var17_hace3': 'AVGSTD', 'ind_var10cte_ult1': 'AVGSTD', 'ind_var31_0': 'AVGSTD', 'num_var22_ult1': 'AVGSTD', 'num_var22_ult3': 'AVGSTD', 'saldo_medio_var12_ult3': 'AVGSTD', 'num_var20': 'AVGSTD', 'imp_compra_var44_hace3': 'AVGSTD', 'imp_sal_var16_ult1': 'AVGSTD', 'num_var25': 'AVGSTD', 'num_var24': 'AVGSTD', 'saldo_medio_var12_ult1': 'AVGSTD', 'num_var26': 'AVGSTD', 'num_var44_0': 'AVGSTD', 'ind_var6_0': 'AVGSTD', 'imp_aport_var13_ult1': 'AVGSTD', 'delta_imp_aport_var17_1y3': 'AVGSTD', 'num_meses_var13_largo_ult3': 'AVGSTD', 'saldo_medio_var13_corto_ult3': 'AVGSTD', 'imp_reemb_var13_ult1': 'AVGSTD', 'saldo_medio_var13_corto_ult1': 'AVGSTD', 'ind_var5_0': 'AVGSTD', 'num_var29_0': 'AVGSTD', 'num_var12': 'AVGSTD', 'num_var14': 'AVGSTD', 'num_var13': 'AVGSTD', 'num_var32_0': 'AVGSTD', 'num_var17': 'AVGSTD', 'ind_var43_recib_ult1': 'AVGSTD', 'num_trasp_var11_ult1': 'AVGSTD', 'ind_var13_corto_0': 'AVGSTD', 'num_op_var40_efect_ult3': 'AVGSTD', 'delta_imp_reemb_var13_1y3': 'AVGSTD', 'num_var40': 'AVGSTD', 'num_var42': 'AVGSTD', 'imp_aport_var33_ult1': 'AVGSTD', 'num_var17_0': 'AVGSTD', 'num_var44': 'AVGSTD', 'ind_var44_0': 'AVGSTD', 'ind_var29_0': 'AVGSTD', 'num_var20_0': 'AVGSTD', 'saldo_var13_largo': 'AVGSTD', 'imp_aport_var33_hace3': 'AVGSTD', 'var3': 'AVGSTD', 'num_med_var22_ult3': 'AVGSTD', 'num_var13_corto': 'AVGSTD', 'imp_op_var40_comer_ult3': 'AVGSTD', 'num_op_var40_ult1': 'AVGSTD', 'imp_op_var40_comer_ult1': 'AVGSTD', 'num_op_var40_ult3': 'AVGSTD', 'ind_var13_largo_0': 'AVGSTD', 'delta_imp_aport_var33_1y3': 'AVGSTD', 'delta_num_aport_var13_1y3': 'AVGSTD', 'saldo_medio_var17_hace2': 'AVGSTD', 'num_var30': 'AVGSTD', 'num_var32': 'AVGSTD', 'num_var31': 'AVGSTD', 'num_var33': 'AVGSTD', 'num_var31_0': 'AVGSTD', 'num_var35': 'AVGSTD', 'num_meses_var17_ult3': 'AVGSTD', 'ind_var33_0': 'AVGSTD', 'num_var37': 'AVGSTD', 'num_var39': 'AVGSTD', 'num_var1_0': 'AVGSTD', 'imp_var43_emit_ult1': 'AVGSTD'} 
        res_ft = obtener_lista_rescalado_para(ml_dataset.columns, "AVGSTD", [TARGET_NAME])

        # Preprocesamos los datos
        preprocesar_datos(ml_dataset, drop_ft, imput_ft, res_ft)

    print("---- Dataset a clasificar:")
    print(ml_dataset.head(5))

    # Si la variable TARGET_NAME está en blanco, suponemos que no tenemos la clase objetivo en los datos
    # Por ello, solo realizaremos predicciones sin comprobar si son correctas o no
    if TARGET_NAME == "":
        # Suponemos que todos los atrobutos pertenecen al conjunto TestX
        print("---- Realizando predicciones sin TestY")
        testX = ml_dataset
        predictions = clf.predict(testX)
        probas = clf.predict_proba(testX)
        
        print("---- Resultados predicciones: ")
        print(predictions)
    
    else:
        print("---- Tratamos el TARGET: " + TARGET_NAME)    
        # Creamos la columna __target__ con el atributo a predecir
        target_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        ml_dataset['__target__'] = ml_dataset[TARGET_NAME].map(str).map(target_map)
        print(ml_dataset)
        testY = ml_dataset[['__target__']].copy() # Creamos testY con las clasificaciones luego
        del ml_dataset['__target__']
        del ml_dataset[TARGET_NAME] # Borramos las clasificaciones del dataset -> testX

        testX = ml_dataset

        print(f"testX -> len: {len(testX.index)}")
        print(testX.head(5))
        print(f"testY -> len: {len(testY.index)}")
        print(testY.head(5))

        print("---- Realizando predicciones")
        predictions = clf.predict(testX)
        probas = clf.predict_proba(testX)

       
        predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
        cols = [
            u'probability_of_value_%s' % label
            for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
        ]
        
        report = classification_report(testY,predictions)
        testY['preds'] = predictions

        print("---- Resultados predicciones: ")
        print(testY)
        print(report)

if __name__ == "__main__":    
    try:
        # options: registra los argumentos del usuario
        # remainder: registra los campos adicionales introducidos -> entrenar_knn.py esto_es_remainder
        options, remainder = getopt(argv[1:], 'ho:i:m:t:', ['help', 'output', 'input', 'model', 'target', 'no-preprocesing'])
        
    except getopt.GetoptError as err:
        # Error al parsear las opciones del comando
        print("ERROR: ", err)
        exit(1)

    print(options)
    # Registramos la configuración del script
    load_options(options)
    # Imprimimos la configuración del script
    show_script_options()
    # Ejecutamos el programa principal
    main()
    

    
