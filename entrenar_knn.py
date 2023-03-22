# Alan García Justel
# Sistemas de apoyo a la decisión
# 3/3/2023

# Imports del script
import os
from getopt import getopt
from sys import exit, argv, version_info
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pickle

# Variables globales
OUTPUT_PATH     = "./models"                    # Path de los archivos de salida
INPUT_FILE      = "SantanderTraHalfHalf.csv"    # Path del archivo de entrada
TARGET_NAME     = "TARGET"                      # Nombre de la columna a clasificar
K               = [1, 3, 5]                     # Lista de numeros minimos de "nearest neighbors"
D               = ['uniform', 'distance']       # Lista de ponderacion de distancias a utilizar
P               = [1, 2]                        # Lista de tipos de distancia -> 1: Manhatan | 2: Euclídea

DEV_SIZE        = 0.2                           # Indice del tamaño del dev. Por defecto un 20% de la muestra
RANDOM_STATE    = 42                            # Seed del random split
PREPROCESADO    = True                          # ¿Preprocesamos los datos de entrada? -> Cat2Num, Missing values y Escalado
BONANZA         = "weighted f1-score"           # Medición utilizada para determinar el mejor modelo -> weighted f1-score | macro f1-score

#######################################################################################
#                              ARGUMENTS AND OPTIONS                                  #
#######################################################################################
def usage():
    # PRE: ---
    # POST: se imprime por pantalla la ayuda del script y salimos del programa
    print("Usage: entrenar_knn.py <optional-args>")
    print("The options supported by entrenar_knn are:")
    print(f"-d              list of distance parameters ->              DEFAULT: {D}")
    print(f"-h, --help      show the usage")
    print(f"-i, --input     input file path of the data                 DEFAULT: ./{INPUT_FILE}")
    print(f"-k              list of neighbors for the KNN algorithm     DEFAULT: {K}")
    print(f"-o, --output    output file path for the weights            DEFAULT: ./{OUTPUT_PATH}")
    print(f"-p              distance from -> 1: Manhatan | 2: Euclidean DEFAULT: {P}")
    print(f"Example: entrenar_knn.py -k 1,3 -p 2 -d uniform")
    print("")

    # Salimos del programa
    exit(1)

def load_options(options):
    # PRE: argumentos especificados por el usuario
    # POST: registramos la configuración del usuario en las variables globales
    global INPUT_FILE, OUTPUT_PATH, K, D, P

    for opt,arg in options:
        if opt == "-d":
            D = arg.split(",")
        elif opt in ('-h', '--help'):
            usage()
        elif opt in ('-i', '--input'):
            INPUT_FILE = str(arg)
        elif opt == '-k':
            K = list(map(lambda n: int(n), arg.split(",")))
        elif opt in ('-o', '--output'):
            OUTPUT_PATH = str(arg)
        elif opt == '-p':
            P = list(map(lambda n: int(n), arg.split(",")))

def show_script_options():
    # PRE: ---
    # POST: imprimimos las configuración del script
    print("entrenar_knn.py configuration:")
    print(f"-d distance parameter   -> {D}")
    print(f"-i input file path      -> {INPUT_FILE}")
    print(f"-k number of neighbors  -> {K}")
    print(f"-o output file path     -> {OUTPUT_PATH}")
    print(f"-p distance algorithm   -> {P}")
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

def imprimir_atributos(atributos):
    # PRE: lista de atributos
    # POST: se imprime por pantalla la lista
    string = ""
    for atr in atributos:
        string += str(f"{atr} ")
    print("---- Atributos seleccionados")
    print(string)
    print()

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
    # POST: Devuelve dos conjuntos: Train y Dev tratando los missing values y reescalados

    # Dividimos nuestros datos de entrenamiento en train y dev
    train, dev = train_test_split(dataset,test_size=DEV_SIZE,random_state=RANDOM_STATE,stratify=dataset[['__target__']])

    # Borramos las instancias que tengan missing values
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        dev = dev[dev[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Inputamos los datos faltantes en función del método elegido en la variable inpute_when_missing
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v1 = train[feature['feature']].mean()
            v2 = dev[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v1 = train[feature['feature']].median()
            v2 = dev[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v1 = 'NULL_CATEGORY'
            v2 = v1
        elif feature['impute_with'] == 'MODE':
            v1 = train[feature['feature']].value_counts().index[0]
            v2 = dev[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v1 = feature['value']
            v2 = v1
        train[feature['feature']] = train[feature['feature']].fillna(v1)
        dev[feature['feature']] = dev[feature['feature']].fillna(v2)
        
        s1 = f"- Train feature {feature['feature']} with value {str(v1)}"
        s2 = f"- Dev feature {feature['feature']} with value {str(v2)}"
        print("Imputed missing values\t%s\t%s" % (s1, s2))
    

    # Reescalamos los datos
    for (feature_name, rescale_method) in rescale_features.items():
        # Obtenemos los valores de rescalado de dev y test
        if rescale_method == 'MINMAX':
            # Valores del train
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale1 = _max - _min
            shift1 = _min
            # Valores del dev
            _min = dev[feature_name].min()
            _max = dev[feature_name].max()
            scale2 = _max - _min
            shift2 = _min

        else:
            # Valores del train
            scale1 = train[feature_name].std()
            shift1 = train[feature_name].mean()
            # Valores del dev
            scale2 = dev[feature_name].std()
            shift2 = dev[feature_name].mean()
        
        # Rescalamos dev y test
        if scale1 == 0. or scale2 == 0.:
            del train[feature_name]
            del dev[feature_name]
            print('Feature %s was dropped because it has no variance in train or dev' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift1).astype(np.float64) / scale1
            dev[feature_name] = (dev[feature_name] - shift2).astype(np.float64) / scale2

    return train, dev

def comprobar_modelo(modelo, devX, devY, target_map):
    predictions = modelo.predict(devX)
    probas = modelo.predict_proba(devX)

    predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=devX.index, columns=cols)

    # Build scored dataset
    results_dev = devX.join(predictions, how='left')
    results_dev = results_dev.join(probabilities, how='left')
    results_dev = results_dev.join(dev['__target__'], how='left')
    results_dev = results_dev.rename(columns= {'__target__': 'TARGET'})

    i=0
    for real,pred in zip(devY,predictions):
        print(real,pred)
        i+=1
        if i>5:
            break
        
    print(f1_score(devY, predictions, average=None))
    print(classification_report(devY,predictions))
    print(confusion_matrix(devY, predictions, labels=[1,0]))

def guardar_modelo(nombre, clf):
    file_path = os.path.join(OUTPUT_PATH, nombre)
    saved_model = pickle.dump(clf, open(file_path,'wb')) 
    print(f'Modelo {nombre} guardado correctamente')

def guardar_report(report):
    nombre = f"{report['modelo']}.csv"
    file_path = os.path.join(OUTPUT_PATH, nombre)
    df = pd.DataFrame(report).transpose()
    df.to_csv(file_path, index = True)
    print(f'Report {nombre} guardado correctamente')

def crear_directorio_modelos():
    # PRE: ---
    # POST: Crea el directorio de modelos. Si ya existe, borra su contenido
    
    # Si no existe, creamos el directorio
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Borramos el contenido del directorio
    for filename in os.listdir(OUTPUT_PATH):
        file_path = os.path.join(OUTPUT_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#######################################################################################
#                                    MAIN PROGRAM                                     #
#######################################################################################
def main():
    # Entrada principal del programa
    print("---- Iniciando main...")
    crear_directorio_modelos()

    # Abrimos el fichero de entrada de datos en un dataframe de pandas
    ml_dataset = pd.read_csv(INPUT_FILE)
    #print(ml_dataset.head(5)) # Imprimimos las primeras 5 lineas
    
    # Seleccionamos atributos son relevantes para la clasificación
    #atributos = ml_dataset.columns # Todos los atributos del dataset
    # Unos pocos atributos
    atributos = ['num_var45_ult1', 'num_op_var39_ult1', 'num_op_var40_comer_ult3', 'num_var45_ult3', 'num_aport_var17_ult1', 'delta_imp_reemb_var17_1y3', 'num_compra_var44_hace3', 'ind_var37_cte', 'num_op_var39_ult3', 'ind_var40', 'num_var12_0', 'num_op_var40_comer_ult1', 'ind_var44', 'ind_var8', 'ind_var24_0', 'ind_var5', 'num_op_var41_hace3', 'ind_var1', 'ind_var8_0', 'num_op_var41_efect_ult3', 'num_op_var41_hace2', 'num_op_var39_hace3', 'num_op_var39_hace2', 'num_aport_var13_hace3', 'num_aport_var33_hace3', 'num_meses_var12_ult3', 'num_op_var41_efect_ult1', 'num_var37_med_ult2', 'num_var7_recib_ult1', 'saldo_medio_var33_hace2', 'saldo_medio_var33_hace3', 'saldo_medio_var8_hace3', 'saldo_medio_var8_hace2', 'imp_op_var39_ult1', 'num_ent_var16_ult1', 'delta_imp_venta_var44_1y3', 'imp_op_var39_efect_ult1', 'ind_var13_0', 'ind_var13_corto', 'saldo_medio_var5_ult3', 'imp_op_var39_efect_ult3', 'saldo_medio_var5_ult1', 'num_op_var40_efect_ult1', 'num_var8_0', 'imp_op_var39_comer_ult1', 'num_var13_largo_0', 'imp_op_var39_comer_ult3', 'num_var45_hace3', 'imp_aport_var13_hace3', 'num_var43_emit_ult1', 'num_var45_hace2', 'num_var13_corto_0', 'num_var8', 'num_var4', 'num_var5', 'num_var1', 'ind_var12_0', 'num_op_var40_hace2', 'num_var33_0', 'ind_var9_cte_ult1', 'imp_op_var40_ult1', 'TARGET', 'num_meses_var39_vig_ult3', 'num_var14_0', 'ind_var10_ult1', 'num_var37_0', 'num_var13_largo', 'delta_imp_aport_var13_1y3', 'saldo_medio_var12_hace3', 'ind_var26_0', 'saldo_medio_var12_hace2', 'num_var40_0', 'ind_var41_0', 'ind_var14', 'ind_var12', 'ind_var13', 'ind_var19', 'ind_var26_cte', 'ind_var17', 'ind_var1_0', 'num_var25_0', 'ind_var43_emit_ult1', 'num_var22_hace2', 'num_var22_hace3', 'saldo_var13', 'saldo_var12', 'num_var6_0', 'saldo_var14', 'saldo_var17', 'imp_op_var41_efect_ult3', 'ind_var32_cte', 'imp_op_var41_efect_ult1', 'ind_var30_0', 'ind_var25', 'ind_var26', 'imp_trans_var37_ult1', 'num_meses_var33_ult3', 'ind_var24', 'imp_var7_recib_ult1', 'imp_ent_var16_ult1', 'imp_aport_var17_hace3', 'num_med_var45_ult3', 'num_var13_0', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'saldo_var20', 'imp_aport_var17_ult1', 'ind_var20', 'ind_var25_0', 'saldo_var24', 'saldo_var26', 'saldo_var25', 'num_op_var41_comer_ult3', 'num_op_var41_comer_ult1', 'ind_var40_0', 'ind_var37', 'ind_var39', 'ind_var25_cte', 'num_var24_0', 'delta_imp_compra_var44_1y3', 'num_aport_var13_ult1', 'ind_var32', 'num_reemb_var13_ult1', 'saldo_medio_var33_ult3', 'ind_var33', 'num_venta_var44_ult1', 'ind_var30', 'saldo_var31', 'ind_var31', 'saldo_var30', 'saldo_var33', 'saldo_var32', 'ind_var14_0', 'saldo_medio_var33_ult1', 'num_var5_0', 'saldo_var37', 'ind_var37_0', 'ind_var13_largo', 'saldo_var13_corto', 'num_meses_var44_ult3', 'num_var39_0', 'num_var43_recib_ult1', 'var21', 'saldo_var40', 'saldo_medio_var17_ult1', 'saldo_var42', 'saldo_medio_var17_ult3', 'saldo_var44', 'num_var42_0', 'delta_num_reemb_var13_1y3', 'saldo_medio_var13_largo_ult1', 'num_op_var39_comer_ult3', 'num_op_var39_comer_ult1', 'ind_var20_0', 'num_op_var41_ult1', 'num_reemb_var17_ult1', 'saldo_medio_var13_largo_ult3', 'num_compra_var44_ult1', 'num_op_var41_ult3', 'num_meses_var29_ult3', 'imp_op_var41_ult1', 'ind_var9_ult1', 'delta_num_reemb_var17_1y3', 'var15', 'imp_compra_var44_ult1', 'imp_op_var40_efect_ult3', 'imp_op_var40_efect_ult1', 'num_var30_0', 'saldo_var5', 'saldo_var8', 'delta_num_aport_var17_1y3', 'saldo_medio_var8_ult3', 'saldo_var1', 'ind_var17_0', 'num_aport_var33_ult1', 'saldo_medio_var13_corto_hace2', 'ind_var32_0', 'imp_venta_var44_ult1', 'saldo_medio_var5_hace2', 'saldo_medio_var13_corto_hace3', 'saldo_medio_var5_hace3', 'delta_num_compra_var44_1y3', 'saldo_medio_var44_hace3', 'ind_var7_recib_ult1', 'saldo_medio_var44_hace2', 'saldo_medio_var8_ult1', 'delta_num_aport_var33_1y3', 'num_var41_0', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'saldo_medio_var13_largo_hace3', 'num_meses_var13_corto_ult3', 'saldo_medio_var13_largo_hace2', 'delta_num_venta_var44_1y3', 'var38', 'num_meses_var5_ult3', 'num_meses_var8_ult3', 'var36', 'num_sal_var16_ult1', 'num_var26_0', 'saldo_medio_var44_ult3', 'ind_var39_0', 'saldo_medio_var44_ult1', 'num_aport_var17_hace3', 'ind_var10cte_ult1', 'ind_var31_0', 'num_var22_ult1', 'num_var22_ult3', 'saldo_medio_var12_ult3', 'num_var20', 'imp_compra_var44_hace3', 'imp_sal_var16_ult1', 'num_var25', 'num_var24', 'saldo_medio_var12_ult1', 'num_var26', 'num_var44_0', 'ind_var6_0', 'imp_aport_var13_ult1', 'delta_imp_aport_var17_1y3', 'num_meses_var13_largo_ult3', 'saldo_medio_var13_corto_ult3', 'imp_reemb_var13_ult1', 'saldo_medio_var13_corto_ult1', 'ind_var5_0', 'num_var29_0', 'num_var12', 'num_var14', 'num_var13', 'num_var32_0', 'num_var17', 'ind_var43_recib_ult1', 'num_trasp_var11_ult1', 'ind_var13_corto_0', 'num_op_var40_efect_ult3', 'delta_imp_reemb_var13_1y3', 'num_var40', 'num_var42', 'imp_aport_var33_ult1', 'num_var17_0', 'num_var44', 'ind_var44_0', 'ind_var29_0', 'num_var20_0', 'saldo_var13_largo', 'imp_aport_var33_hace3', 'var3', 'num_med_var22_ult3', 'num_var13_corto', 'imp_op_var40_comer_ult3', 'num_op_var40_ult1', 'imp_op_var40_comer_ult1', 'num_op_var40_ult3', 'ind_var13_largo_0', 'delta_imp_aport_var33_1y3', 'delta_num_aport_var13_1y3', 'saldo_medio_var17_hace2', 'num_var30', 'num_var32', 'num_var31', 'num_var33', 'num_var31_0', 'num_var35', 'num_meses_var17_ult3', 'ind_var33_0', 'num_var37', 'num_var39', 'num_var1_0', 'imp_var43_emit_ult1']
    #atributos = atributos_excepto(atributos, []) # Todos menos...
    
    imprimir_atributos(atributos) # Mostramos los atributos elegidos

    # De todo el conjunto de datos nos quedamos con aquellos atributos relevantes
    ml_dataset = ml_dataset[atributos]

    print("---- Estandarizamos en Unicode y pasamos de atributos categoricos a numericos")
    categorical_features = [] # TODO: Detectar variables categóricas -> pd.get_dummies...
    text_features = []
    numerical_features = atributos_excepto(ml_dataset.columns, [TARGET_NAME] + categorical_features + text_features)
    
    # Ponemos los datos en un formato común
    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)
    
    print("---- Tratamos el TARGET: " + TARGET_NAME)    
    # Creamos la columna __target__ con el atributo a predecir
    target_map = {'0': 0, '1': 1}
    ml_dataset['__target__'] = ml_dataset[TARGET_NAME].map(str).map(target_map)
    ml_dataset['__target__'] = ml_dataset[TARGET_NAME]
    del ml_dataset[TARGET_NAME]
    
    # Borramos aquellas entradas de datos en las que el target sea null
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

    # Convertir variables categoricas en numericas
    ml_dataset[categorical_features] = ml_dataset[categorical_features].apply(lambda x: pd.factorize(x)[0])
    
    print("---- Dataset empleado")
    print(ml_dataset.head(5)) # Imprimimos las primeras 5 lineas

    print("---- Preprocesamos los datos")
    
    # Definimos los parámetros del preprocesado
    # TODO: Establecer imputacion con media a todos los atributos por defecto y reescalado con AVGSTD
    drop_ft = [] # ['num_var45_ult1', ...] array
    # [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'}, ...] array de diccionarios
    imput_ft = [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var45_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_reemb_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_compra_var44_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var37_cte', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var40', 'impute_with': 'MEAN'}, {'feature': 'num_var12_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var44', 'impute_with': 'MEAN'}, {'feature': 'ind_var8', 'impute_with': 'MEAN'}, {'feature': 'ind_var24_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var5', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var1', 'impute_with': 'MEAN'}, {'feature': 'ind_var8_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var13_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var33_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var12_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var37_med_ult2', 'impute_with': 'MEAN'}, {'feature': 'num_var7_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_hace2', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_hace2', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_ent_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_venta_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_corto', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var8_0', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var13_largo_0', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var39_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var45_hace3', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var13_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_var43_emit_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var45_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var13_corto_0', 'impute_with': 'MEAN'}, {'feature': 'num_var8', 'impute_with': 'MEAN'}, {'feature': 'num_var4', 'impute_with': 'MEAN'}, {'feature': 'num_var5', 'impute_with': 'MEAN'}, {'feature': 'num_var1', 'impute_with': 'MEAN'}, {'feature': 'ind_var12_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var33_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var9_cte_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var39_vig_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var14_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var10_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var37_0', 'impute_with': 'MEAN'}, {'feature': 'num_var13_largo', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_aport_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var26_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var40_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var41_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var14', 'impute_with': 'MEAN'}, {'feature': 'ind_var12', 'impute_with': 'MEAN'}, {'feature': 'ind_var13', 'impute_with': 'MEAN'}, {'feature': 'ind_var19', 'impute_with': 'MEAN'}, {'feature': 'ind_var26_cte', 'impute_with': 'MEAN'}, {'feature': 'ind_var17', 'impute_with': 'MEAN'}, {'feature': 'ind_var1_0', 'impute_with': 'MEAN'}, {'feature': 'num_var25_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var43_emit_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var22_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var22_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var13', 'impute_with': 'MEAN'}, {'feature': 'saldo_var12', 'impute_with': 'MEAN'}, {'feature': 'num_var6_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var14', 'impute_with': 'MEAN'}, {'feature': 'saldo_var17', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var32_cte', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var30_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var25', 'impute_with': 'MEAN'}, {'feature': 'ind_var26', 'impute_with': 'MEAN'}, {'feature': 'imp_trans_var37_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var33_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var24', 'impute_with': 'MEAN'}, {'feature': 'imp_var7_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_ent_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var17_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_med_var45_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var13_0', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var20', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var20', 'impute_with': 'MEAN'}, {'feature': 'ind_var25_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var24', 'impute_with': 'MEAN'}, {'feature': 'saldo_var26', 'impute_with': 'MEAN'}, {'feature': 'saldo_var25', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var40_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var37', 'impute_with': 'MEAN'}, {'feature': 'ind_var39', 'impute_with': 'MEAN'}, {'feature': 'ind_var25_cte', 'impute_with': 'MEAN'}, {'feature': 'num_var24_0', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_compra_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var32', 'impute_with': 'MEAN'}, {'feature': 'num_reemb_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var33', 'impute_with': 'MEAN'}, {'feature': 'num_venta_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var30', 'impute_with': 'MEAN'}, {'feature': 'saldo_var31', 'impute_with': 'MEAN'}, {'feature': 'ind_var31', 'impute_with': 'MEAN'}, {'feature': 'saldo_var30', 'impute_with': 'MEAN'}, {'feature': 'saldo_var33', 'impute_with': 'MEAN'}, {'feature': 'saldo_var32', 'impute_with': 'MEAN'}, {'feature': 'ind_var14_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var33_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var5_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var37', 'impute_with': 'MEAN'}, {'feature': 'ind_var37_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_largo', 'impute_with': 'MEAN'}, {'feature': 'saldo_var13_corto', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var44_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var39_0', 'impute_with': 'MEAN'}, {'feature': 'num_var43_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'var21', 'impute_with': 'MEAN'}, {'feature': 'saldo_var40', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_var42', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var17_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var44', 'impute_with': 'MEAN'}, {'feature': 'num_var42_0', 'impute_with': 'MEAN'}, {'feature': 'delta_num_reemb_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var20_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_reemb_var17_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_compra_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var41_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var29_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var41_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var9_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_num_reemb_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'var15', 'impute_with': 'MEAN'}, {'feature': 'imp_compra_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var30_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var5', 'impute_with': 'MEAN'}, {'feature': 'saldo_var8', 'impute_with': 'MEAN'}, {'feature': 'delta_num_aport_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_var1', 'impute_with': 'MEAN'}, {'feature': 'ind_var17_0', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var33_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_hace2', 'impute_with': 'MEAN'}, {'feature': 'ind_var32_0', 'impute_with': 'MEAN'}, {'feature': 'imp_venta_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_hace2', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_hace3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var5_hace3', 'impute_with': 'MEAN'}, {'feature': 'delta_num_compra_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var7_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_hace2', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var8_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_num_aport_var33_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_var41_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_efect_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var39_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_hace3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var13_corto_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_largo_hace2', 'impute_with': 'MEAN'}, {'feature': 'delta_num_venta_var44_1y3', 'impute_with': 'MEAN'}, {'feature': 'var38', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var5_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var8_ult3', 'impute_with': 'MEAN'}, {'feature': 'var36', 'impute_with': 'MEAN'}, {'feature': 'num_sal_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var26_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var39_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var44_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_aport_var17_hace3', 'impute_with': 'MEAN'}, {'feature': 'ind_var10cte_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var31_0', 'impute_with': 'MEAN'}, {'feature': 'num_var22_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var22_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var20', 'impute_with': 'MEAN'}, {'feature': 'imp_compra_var44_hace3', 'impute_with': 'MEAN'}, {'feature': 'imp_sal_var16_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var25', 'impute_with': 'MEAN'}, {'feature': 'num_var24', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var12_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var26', 'impute_with': 'MEAN'}, {'feature': 'num_var44_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var6_0', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_aport_var17_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var13_largo_ult3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_ult3', 'impute_with': 'MEAN'}, {'feature': 'imp_reemb_var13_ult1', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var13_corto_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var5_0', 'impute_with': 'MEAN'}, {'feature': 'num_var29_0', 'impute_with': 'MEAN'}, {'feature': 'num_var12', 'impute_with': 'MEAN'}, {'feature': 'num_var14', 'impute_with': 'MEAN'}, {'feature': 'num_var13', 'impute_with': 'MEAN'}, {'feature': 'num_var32_0', 'impute_with': 'MEAN'}, {'feature': 'num_var17', 'impute_with': 'MEAN'}, {'feature': 'ind_var43_recib_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_trasp_var11_ult1', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_corto_0', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_efect_ult3', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_reemb_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'num_var40', 'impute_with': 'MEAN'}, {'feature': 'num_var42', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var33_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_var17_0', 'impute_with': 'MEAN'}, {'feature': 'num_var44', 'impute_with': 'MEAN'}, {'feature': 'ind_var44_0', 'impute_with': 'MEAN'}, {'feature': 'ind_var29_0', 'impute_with': 'MEAN'}, {'feature': 'num_var20_0', 'impute_with': 'MEAN'}, {'feature': 'saldo_var13_largo', 'impute_with': 'MEAN'}, {'feature': 'imp_aport_var33_hace3', 'impute_with': 'MEAN'}, {'feature': 'var3', 'impute_with': 'MEAN'}, {'feature': 'num_med_var22_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_var13_corto', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_comer_ult3', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_ult1', 'impute_with': 'MEAN'}, {'feature': 'imp_op_var40_comer_ult1', 'impute_with': 'MEAN'}, {'feature': 'num_op_var40_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var13_largo_0', 'impute_with': 'MEAN'}, {'feature': 'delta_imp_aport_var33_1y3', 'impute_with': 'MEAN'}, {'feature': 'delta_num_aport_var13_1y3', 'impute_with': 'MEAN'}, {'feature': 'saldo_medio_var17_hace2', 'impute_with': 'MEAN'}, {'feature': 'num_var30', 'impute_with': 'MEAN'}, {'feature': 'num_var32', 'impute_with': 'MEAN'}, {'feature': 'num_var31', 'impute_with': 'MEAN'}, {'feature': 'num_var33', 'impute_with': 'MEAN'}, {'feature': 'num_var31_0', 'impute_with': 'MEAN'}, {'feature': 'num_var35', 'impute_with': 'MEAN'}, {'feature': 'num_meses_var17_ult3', 'impute_with': 'MEAN'}, {'feature': 'ind_var33_0', 'impute_with': 'MEAN'}, {'feature': 'num_var37', 'impute_with': 'MEAN'}, {'feature': 'num_var39', 'impute_with': 'MEAN'}, {'feature': 'num_var1_0', 'impute_with': 'MEAN'}, {'feature': 'imp_var43_emit_ult1', 'impute_with': 'MEAN'}] 
    #imput_ft = obtener_lista_impute_para(ml_dataset.columns, "MEAN", ["__target__"])
    # {'num_var45_ult1': 'AVGSTD', ... } diccionario
    res_ft = {'num_var45_ult1': 'AVGSTD', 'num_op_var39_ult1': 'AVGSTD', 'num_op_var40_comer_ult3': 'AVGSTD', 'num_var45_ult3': 'AVGSTD', 'num_aport_var17_ult1': 'AVGSTD', 'delta_imp_reemb_var17_1y3': 'AVGSTD', 'num_compra_var44_hace3': 'AVGSTD', 'ind_var37_cte': 'AVGSTD', 'num_op_var39_ult3': 'AVGSTD', 'ind_var40': 'AVGSTD', 'num_var12_0': 'AVGSTD', 'num_op_var40_comer_ult1': 'AVGSTD', 'ind_var44': 'AVGSTD', 'ind_var8': 'AVGSTD', 'ind_var24_0': 'AVGSTD', 'ind_var5': 'AVGSTD', 'num_op_var41_hace3': 'AVGSTD', 'ind_var1': 'AVGSTD', 'ind_var8_0': 'AVGSTD', 'num_op_var41_efect_ult3': 'AVGSTD', 'num_op_var41_hace2': 'AVGSTD', 'num_op_var39_hace3': 'AVGSTD', 'num_op_var39_hace2': 'AVGSTD', 'num_aport_var13_hace3': 'AVGSTD', 'num_aport_var33_hace3': 'AVGSTD', 'num_meses_var12_ult3': 'AVGSTD', 'num_op_var41_efect_ult1': 'AVGSTD', 'num_var37_med_ult2': 'AVGSTD', 'num_var7_recib_ult1': 'AVGSTD', 'saldo_medio_var33_hace2': 'AVGSTD', 'saldo_medio_var33_hace3': 'AVGSTD', 'saldo_medio_var8_hace3': 'AVGSTD', 'saldo_medio_var8_hace2': 'AVGSTD', 'imp_op_var39_ult1': 'AVGSTD', 'num_ent_var16_ult1': 'AVGSTD', 'delta_imp_venta_var44_1y3': 'AVGSTD', 'imp_op_var39_efect_ult1': 'AVGSTD', 'ind_var13_0': 'AVGSTD', 'ind_var13_corto': 'AVGSTD', 'saldo_medio_var5_ult3': 'AVGSTD', 'imp_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var5_ult1': 'AVGSTD', 'num_op_var40_efect_ult1': 'AVGSTD', 'num_var8_0': 'AVGSTD', 'imp_op_var39_comer_ult1': 'AVGSTD', 'num_var13_largo_0': 'AVGSTD', 'imp_op_var39_comer_ult3': 'AVGSTD', 'num_var45_hace3': 'AVGSTD', 'imp_aport_var13_hace3': 'AVGSTD', 'num_var43_emit_ult1': 'AVGSTD', 'num_var45_hace2': 'AVGSTD', 'num_var13_corto_0': 'AVGSTD', 'num_var8': 'AVGSTD', 'num_var4': 'AVGSTD', 'num_var5': 'AVGSTD', 'num_var1': 'AVGSTD', 'ind_var12_0': 'AVGSTD', 'num_op_var40_hace2': 'AVGSTD', 'num_var33_0': 'AVGSTD', 'ind_var9_cte_ult1': 'AVGSTD', 'imp_op_var40_ult1': 'AVGSTD', 'num_meses_var39_vig_ult3': 'AVGSTD', 'num_var14_0': 'AVGSTD', 'ind_var10_ult1': 'AVGSTD', 'num_var37_0': 'AVGSTD', 'num_var13_largo': 'AVGSTD', 'delta_imp_aport_var13_1y3': 'AVGSTD', 'saldo_medio_var12_hace3': 'AVGSTD', 'ind_var26_0': 'AVGSTD', 'saldo_medio_var12_hace2': 'AVGSTD', 'num_var40_0': 'AVGSTD', 'ind_var41_0': 'AVGSTD', 'ind_var14': 'AVGSTD', 'ind_var12': 'AVGSTD', 'ind_var13': 'AVGSTD', 'ind_var19': 'AVGSTD', 'ind_var26_cte': 'AVGSTD', 'ind_var17': 'AVGSTD', 'ind_var1_0': 'AVGSTD', 'num_var25_0': 'AVGSTD', 'ind_var43_emit_ult1': 'AVGSTD', 'num_var22_hace2': 'AVGSTD', 'num_var22_hace3': 'AVGSTD', 'saldo_var13': 'AVGSTD', 'saldo_var12': 'AVGSTD', 'num_var6_0': 'AVGSTD', 'saldo_var14': 'AVGSTD', 'saldo_var17': 'AVGSTD', 'imp_op_var41_efect_ult3': 'AVGSTD', 'ind_var32_cte': 'AVGSTD', 'imp_op_var41_efect_ult1': 'AVGSTD', 'ind_var30_0': 'AVGSTD', 'ind_var25': 'AVGSTD', 'ind_var26': 'AVGSTD', 'imp_trans_var37_ult1': 'AVGSTD', 'num_meses_var33_ult3': 'AVGSTD', 'ind_var24': 'AVGSTD', 'imp_var7_recib_ult1': 'AVGSTD', 'imp_ent_var16_ult1': 'AVGSTD', 'imp_aport_var17_hace3': 'AVGSTD', 'num_med_var45_ult3': 'AVGSTD', 'num_var13_0': 'AVGSTD', 'imp_op_var41_comer_ult1': 'AVGSTD', 'imp_op_var41_comer_ult3': 'AVGSTD', 'saldo_var20': 'AVGSTD', 'imp_aport_var17_ult1': 'AVGSTD', 'ind_var20': 'AVGSTD', 'ind_var25_0': 'AVGSTD', 'saldo_var24': 'AVGSTD', 'saldo_var26': 'AVGSTD', 'saldo_var25': 'AVGSTD', 'num_op_var41_comer_ult3': 'AVGSTD', 'num_op_var41_comer_ult1': 'AVGSTD', 'ind_var40_0': 'AVGSTD', 'ind_var37': 'AVGSTD', 'ind_var39': 'AVGSTD', 'ind_var25_cte': 'AVGSTD', 'num_var24_0': 'AVGSTD', 'delta_imp_compra_var44_1y3': 'AVGSTD', 'num_aport_var13_ult1': 'AVGSTD', 'ind_var32': 'AVGSTD', 'num_reemb_var13_ult1': 'AVGSTD', 'saldo_medio_var33_ult3': 'AVGSTD', 'ind_var33': 'AVGSTD', 'num_venta_var44_ult1': 'AVGSTD', 'ind_var30': 'AVGSTD', 'saldo_var31': 'AVGSTD', 'ind_var31': 'AVGSTD', 'saldo_var30': 'AVGSTD', 'saldo_var33': 'AVGSTD', 'saldo_var32': 'AVGSTD', 'ind_var14_0': 'AVGSTD', 'saldo_medio_var33_ult1': 'AVGSTD', 'num_var5_0': 'AVGSTD', 'saldo_var37': 'AVGSTD', 'ind_var37_0': 'AVGSTD', 'ind_var13_largo': 'AVGSTD', 'saldo_var13_corto': 'AVGSTD', 'num_meses_var44_ult3': 'AVGSTD', 'num_var39_0': 'AVGSTD', 'num_var43_recib_ult1': 'AVGSTD', 'var21': 'AVGSTD', 'saldo_var40': 'AVGSTD', 'saldo_medio_var17_ult1': 'AVGSTD', 'saldo_var42': 'AVGSTD', 'saldo_medio_var17_ult3': 'AVGSTD', 'saldo_var44': 'AVGSTD', 'num_var42_0': 'AVGSTD', 'delta_num_reemb_var13_1y3': 'AVGSTD', 'saldo_medio_var13_largo_ult1': 'AVGSTD', 'num_op_var39_comer_ult3': 'AVGSTD', 'num_op_var39_comer_ult1': 'AVGSTD', 'ind_var20_0': 'AVGSTD', 'num_op_var41_ult1': 'AVGSTD', 'num_reemb_var17_ult1': 'AVGSTD', 'saldo_medio_var13_largo_ult3': 'AVGSTD', 'num_compra_var44_ult1': 'AVGSTD', 'num_op_var41_ult3': 'AVGSTD', 'num_meses_var29_ult3': 'AVGSTD', 'imp_op_var41_ult1': 'AVGSTD', 'ind_var9_ult1': 'AVGSTD', 'delta_num_reemb_var17_1y3': 'AVGSTD', 'var15': 'AVGSTD', 'imp_compra_var44_ult1': 'AVGSTD', 'imp_op_var40_efect_ult3': 'AVGSTD', 'imp_op_var40_efect_ult1': 'AVGSTD', 'num_var30_0': 'AVGSTD', 'saldo_var5': 'AVGSTD', 'saldo_var8': 'AVGSTD', 'delta_num_aport_var17_1y3': 'AVGSTD', 'saldo_medio_var8_ult3': 'AVGSTD', 'saldo_var1': 'AVGSTD', 'ind_var17_0': 'AVGSTD', 'num_aport_var33_ult1': 'AVGSTD', 'saldo_medio_var13_corto_hace2': 'AVGSTD', 'ind_var32_0': 'AVGSTD', 'imp_venta_var44_ult1': 'AVGSTD', 'saldo_medio_var5_hace2': 'AVGSTD', 'saldo_medio_var13_corto_hace3': 'AVGSTD', 'saldo_medio_var5_hace3': 'AVGSTD', 'delta_num_compra_var44_1y3': 'AVGSTD', 'saldo_medio_var44_hace3': 'AVGSTD', 'ind_var7_recib_ult1': 'AVGSTD', 'saldo_medio_var44_hace2': 'AVGSTD', 'saldo_medio_var8_ult1': 'AVGSTD', 'delta_num_aport_var33_1y3': 'AVGSTD', 'num_var41_0': 'AVGSTD', 'num_op_var39_efect_ult1': 'AVGSTD', 'num_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace3': 'AVGSTD', 'num_meses_var13_corto_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace2': 'AVGSTD', 'delta_num_venta_var44_1y3': 'AVGSTD', 'var38': 'AVGSTD', 'num_meses_var5_ult3': 'AVGSTD', 'num_meses_var8_ult3': 'AVGSTD', 'var36': 'AVGSTD', 'num_sal_var16_ult1': 'AVGSTD', 'num_var26_0': 'AVGSTD', 'saldo_medio_var44_ult3': 'AVGSTD', 'ind_var39_0': 'AVGSTD', 'saldo_medio_var44_ult1': 'AVGSTD', 'num_aport_var17_hace3': 'AVGSTD', 'ind_var10cte_ult1': 'AVGSTD', 'ind_var31_0': 'AVGSTD', 'num_var22_ult1': 'AVGSTD', 'num_var22_ult3': 'AVGSTD', 'saldo_medio_var12_ult3': 'AVGSTD', 'num_var20': 'AVGSTD', 'imp_compra_var44_hace3': 'AVGSTD', 'imp_sal_var16_ult1': 'AVGSTD', 'num_var25': 'AVGSTD', 'num_var24': 'AVGSTD', 'saldo_medio_var12_ult1': 'AVGSTD', 'num_var26': 'AVGSTD', 'num_var44_0': 'AVGSTD', 'ind_var6_0': 'AVGSTD', 'imp_aport_var13_ult1': 'AVGSTD', 'delta_imp_aport_var17_1y3': 'AVGSTD', 'num_meses_var13_largo_ult3': 'AVGSTD', 'saldo_medio_var13_corto_ult3': 'AVGSTD', 'imp_reemb_var13_ult1': 'AVGSTD', 'saldo_medio_var13_corto_ult1': 'AVGSTD', 'ind_var5_0': 'AVGSTD', 'num_var29_0': 'AVGSTD', 'num_var12': 'AVGSTD', 'num_var14': 'AVGSTD', 'num_var13': 'AVGSTD', 'num_var32_0': 'AVGSTD', 'num_var17': 'AVGSTD', 'ind_var43_recib_ult1': 'AVGSTD', 'num_trasp_var11_ult1': 'AVGSTD', 'ind_var13_corto_0': 'AVGSTD', 'num_op_var40_efect_ult3': 'AVGSTD', 'delta_imp_reemb_var13_1y3': 'AVGSTD', 'num_var40': 'AVGSTD', 'num_var42': 'AVGSTD', 'imp_aport_var33_ult1': 'AVGSTD', 'num_var17_0': 'AVGSTD', 'num_var44': 'AVGSTD', 'ind_var44_0': 'AVGSTD', 'ind_var29_0': 'AVGSTD', 'num_var20_0': 'AVGSTD', 'saldo_var13_largo': 'AVGSTD', 'imp_aport_var33_hace3': 'AVGSTD', 'var3': 'AVGSTD', 'num_med_var22_ult3': 'AVGSTD', 'num_var13_corto': 'AVGSTD', 'imp_op_var40_comer_ult3': 'AVGSTD', 'num_op_var40_ult1': 'AVGSTD', 'imp_op_var40_comer_ult1': 'AVGSTD', 'num_op_var40_ult3': 'AVGSTD', 'ind_var13_largo_0': 'AVGSTD', 'delta_imp_aport_var33_1y3': 'AVGSTD', 'delta_num_aport_var13_1y3': 'AVGSTD', 'saldo_medio_var17_hace2': 'AVGSTD', 'num_var30': 'AVGSTD', 'num_var32': 'AVGSTD', 'num_var31': 'AVGSTD', 'num_var33': 'AVGSTD', 'num_var31_0': 'AVGSTD', 'num_var35': 'AVGSTD', 'num_meses_var17_ult3': 'AVGSTD', 'ind_var33_0': 'AVGSTD', 'num_var37': 'AVGSTD', 'num_var39': 'AVGSTD', 'num_var1_0': 'AVGSTD', 'imp_var43_emit_ult1': 'AVGSTD'} 
    #res_ft = obtener_lista_rescalado_para(ml_dataset.columns, "AVGSTD", ["__target__"])

    # preprocesar_datos(dataset, categorical_features, text_features, numerical_features, drop_rows_when_missing, impute_when_missing, rescale_features)
    train, dev = preprocesar_datos(ml_dataset, drop_ft, imput_ft, res_ft)
    # train, dev = emergencia

    print("---- Dataset preprocesado")
    print("TRAIN: ")
    print(train.head(5)) # Imprimimos las primeras 5 lineas
    print(train['__target__'].value_counts())
    print("DEV: ")
    print(dev.head(5))
    print(dev['__target__'].value_counts())

    
    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']
    devX = dev.drop('__target__', axis=1)
    #devY = devt['__target__']
    trainY = np.array(train['__target__'])
    devY = np.array(dev['__target__'])
    
    print("---- Undersampling...")
    # Hacer un undersample para que las clases menos representadas estén más presentes para el entreno
    # sampling_strategy = diccionario {'clase': valor} -> multiclass
    undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces
    
    trainX,trainY = undersample.fit_resample(trainX,trainY)
    devX,devY = undersample.fit_resample(devX, devY)

    print("---- Iniciando barrido de parámetros ")
    print("TRAINX: ")
    print(trainX.head(5)) # Imprimimos las primeras 5 lineas
    print("DEVX: ")
    print(devX.head(5))


    mejor = {'value': 0.0, 'name': 'empty'}
    for w in D:
        for k in K:
            for p in P:
                nombre_modelo = f"KNN-k:{k}-p:{p}-w:{w}"
                print(nombre_modelo)
                # weights : {'uniform', 'distance'} or callable, default='uniform'
                # algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}
                
                clf = KNeighborsClassifier(n_neighbors=k,
                                    weights=w,
                                    algorithm='auto',
                                    leaf_size=30,
                                    p=p)
                # Indica que las clases están balanceadas -> paso anterior de undersampling
                clf.class_weight = "balanced"
                # Ajustamos el modelo a nuestro train
                clf.fit(trainX, trainY)

                # Aplicamos el dev para ver el comportamiento del modelo
                #comprobar_modelo(clf, devX, devY, target_map)

                predictions = clf.predict(devX)
                probas = clf.predict_proba(devX)
                

                predictions = pd.Series(data=predictions, index=devX.index, name='predicted_value')
                
                cols = [
                    u'probability_of_value_%s' % label
                    for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
                ]

                probabilities = pd.DataFrame(data=probas, index=devX.index, columns=cols)

                results_dev = devX.join(predictions, how='left')
                results_dev = results_dev.join(probabilities, how='left')
                results_dev = results_dev.join(dev['__target__'], how='left')
                results_dev = results_dev.rename(columns= {'__target__': 'TARGET'})

                # i=0
                # for real,pred in zip(devY,predictions):
                #     print(real,pred)
                #     i+=1
                #     if i>5:
                #         break

                # f1 = f1_score(devY, predictions, average=None)
                # print(f"F1_score: {f1}")

                report = classification_report(devY,predictions, output_dict=True)
                report['modelo'] = nombre_modelo
                print(report)

                # Medimos la bonanza del modelo
                b = 0.0
                if BONANZA == "weighted f1-score":
                    b = report['weighted avg']['f1-score']
                elif BONANZA == "macro f1-score":
                    b = report['macro avg']['f1-score']

                if b > mejor['value']:
                    mejor['value'] = b
                    mejor['name'] = nombre_modelo

                # print(confusion_matrix(devY, predictions, labels=[1,0]))
                guardar_report(report)
                guardar_modelo(f"{nombre_modelo}.sav", clf)
                print()

    print(f"El mejor modelo según <{BONANZA}> es {mejor}")
    

if __name__ == "__main__":    
    try:
        # options: registra los argumentos del usuario
        # remainder: registra los campos adicionales introducidos -> entrenar_knn.py esto_es_remainder
        options, remainder = getopt(argv[1:], 'd:h:i:k:o:p:', ['help', 'input', 'output'])
    
    except getopt.GetoptError as err:
        # Error al parsear las opciones del comando
        print("ERROR: ", err)
        exit(1)

    # Registramos la configuración del script
    load_options(options)
    # Imprimimos la configuración del script
    show_script_options()
    # Ejecutamos el programa principal
    main()