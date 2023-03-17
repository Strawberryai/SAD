# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import getopt
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

k=1
d=1
p='./'
f="SantanderTraHalfHalf.csv"
oFile="output.out"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('ARGV   :',sys.argv[1:])
    try:
        options,remainder = getopt.getopt(sys.argv[1:],'o:k:d:p:f:h',['output=','k=','d=','path=','iFile','h'])
    except getopt.GetoptError as err:
        print('ERROR:',err)
        sys.exit(1)
    print('OPTIONS   :',options)

    for opt,arg in options:
        if opt in ('-o','--output'):
            oFile = arg
        elif opt == '-k':
            k = arg
        elif opt ==  '-d':
            d = arg
        elif opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-h','--help'):
            print(' -o outputFile \n -k numberOfItems \n -d distanceParameter \n -p inputFilePath \n -f inputFileName \n ')
            exit(1)

    if p == './':
        iFile=p+str(f)
    else:
        iFile = p+"/" + str(f)
    # astype('unicode') does not work as expected

    def coerce_to_unicode(x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x, 'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)

    #Abrir el fichero .csv y cargarlo en un dataframe de pandas
    ml_dataset = pd.read_csv(iFile)

    #comprobar que los datos se han cargado bien. Cuidado con las cabeceras, la primera línea por defecto la considerara como la que almacena los nombres de los atributos
    # comprobar los parametros por defecto del pd.read_csv en lo referente a las cabeceras si no se quiere lo comentado

    #print(ml_dataset.head(5))

    ml_dataset = ml_dataset[
        ['num_var45_ult1', 'num_op_var39_ult1', 'num_op_var40_comer_ult3', 'num_var45_ult3', 'num_aport_var17_ult1',
         'delta_imp_reemb_var17_1y3', 'num_compra_var44_hace3', 'ind_var37_cte', 'num_op_var39_ult3', 'ind_var40',
         'num_var12_0', 'num_op_var40_comer_ult1', 'ind_var44', 'ind_var8', 'ind_var24_0', 'ind_var5',
         'num_op_var41_hace3', 'ind_var1', 'ind_var8_0', 'num_op_var41_efect_ult3', 'num_op_var41_hace2',
         'num_op_var39_hace3', 'num_op_var39_hace2', 'num_aport_var13_hace3', 'num_aport_var33_hace3',
         'num_meses_var12_ult3', 'num_op_var41_efect_ult1', 'num_var37_med_ult2', 'num_var7_recib_ult1',
         'saldo_medio_var33_hace2', 'saldo_medio_var33_hace3', 'saldo_medio_var8_hace3', 'saldo_medio_var8_hace2',
         'imp_op_var39_ult1', 'num_ent_var16_ult1', 'delta_imp_venta_var44_1y3', 'imp_op_var39_efect_ult1',
         'ind_var13_0', 'ind_var13_corto', 'saldo_medio_var5_ult3', 'imp_op_var39_efect_ult3', 'saldo_medio_var5_ult1',
         'num_op_var40_efect_ult1', 'num_var8_0', 'imp_op_var39_comer_ult1', 'num_var13_largo_0',
         'imp_op_var39_comer_ult3', 'num_var45_hace3', 'imp_aport_var13_hace3', 'num_var43_emit_ult1',
         'num_var45_hace2', 'num_var13_corto_0', 'num_var8', 'num_var4', 'num_var5', 'num_var1', 'ind_var12_0',
         'num_op_var40_hace2', 'num_var33_0', 'ind_var9_cte_ult1', 'imp_op_var40_ult1', 'TARGET',
         'num_meses_var39_vig_ult3', 'num_var14_0', 'ind_var10_ult1', 'num_var37_0', 'num_var13_largo',
         'delta_imp_aport_var13_1y3', 'saldo_medio_var12_hace3', 'ind_var26_0', 'saldo_medio_var12_hace2',
         'num_var40_0', 'ind_var41_0', 'ind_var14', 'ind_var12', 'ind_var13', 'ind_var19', 'ind_var26_cte', 'ind_var17',
         'ind_var1_0', 'num_var25_0', 'ind_var43_emit_ult1', 'num_var22_hace2', 'num_var22_hace3', 'saldo_var13',
         'saldo_var12', 'num_var6_0', 'saldo_var14', 'saldo_var17', 'imp_op_var41_efect_ult3', 'ind_var32_cte',
         'imp_op_var41_efect_ult1', 'ind_var30_0', 'ind_var25', 'ind_var26', 'imp_trans_var37_ult1',
         'num_meses_var33_ult3', 'ind_var24', 'imp_var7_recib_ult1', 'imp_ent_var16_ult1', 'imp_aport_var17_hace3',
         'num_med_var45_ult3', 'num_var13_0', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'saldo_var20',
         'imp_aport_var17_ult1', 'ind_var20', 'ind_var25_0', 'saldo_var24', 'saldo_var26', 'saldo_var25',
         'num_op_var41_comer_ult3', 'num_op_var41_comer_ult1', 'ind_var40_0', 'ind_var37', 'ind_var39', 'ind_var25_cte',
         'num_var24_0', 'delta_imp_compra_var44_1y3', 'num_aport_var13_ult1', 'ind_var32', 'num_reemb_var13_ult1',
         'saldo_medio_var33_ult3', 'ind_var33', 'num_venta_var44_ult1', 'ind_var30', 'saldo_var31', 'ind_var31',
         'saldo_var30', 'saldo_var33', 'saldo_var32', 'ind_var14_0', 'saldo_medio_var33_ult1', 'num_var5_0',
         'saldo_var37', 'ind_var37_0', 'ind_var13_largo', 'saldo_var13_corto', 'num_meses_var44_ult3', 'num_var39_0',
         'num_var43_recib_ult1', 'var21', 'saldo_var40', 'saldo_medio_var17_ult1', 'saldo_var42',
         'saldo_medio_var17_ult3', 'saldo_var44', 'num_var42_0', 'delta_num_reemb_var13_1y3',
         'saldo_medio_var13_largo_ult1', 'num_op_var39_comer_ult3', 'num_op_var39_comer_ult1', 'ind_var20_0',
         'num_op_var41_ult1', 'num_reemb_var17_ult1', 'saldo_medio_var13_largo_ult3', 'num_compra_var44_ult1',
         'num_op_var41_ult3', 'num_meses_var29_ult3', 'imp_op_var41_ult1', 'ind_var9_ult1', 'delta_num_reemb_var17_1y3',
         'var15', 'imp_compra_var44_ult1', 'imp_op_var40_efect_ult3', 'imp_op_var40_efect_ult1', 'num_var30_0',
         'saldo_var5', 'saldo_var8', 'delta_num_aport_var17_1y3', 'saldo_medio_var8_ult3', 'saldo_var1', 'ind_var17_0',
         'num_aport_var33_ult1', 'saldo_medio_var13_corto_hace2', 'ind_var32_0', 'imp_venta_var44_ult1',
         'saldo_medio_var5_hace2', 'saldo_medio_var13_corto_hace3', 'saldo_medio_var5_hace3',
         'delta_num_compra_var44_1y3', 'saldo_medio_var44_hace3', 'ind_var7_recib_ult1', 'saldo_medio_var44_hace2',
         'saldo_medio_var8_ult1', 'delta_num_aport_var33_1y3', 'num_var41_0', 'num_op_var39_efect_ult1',
         'num_op_var39_efect_ult3', 'saldo_medio_var13_largo_hace3', 'num_meses_var13_corto_ult3',
         'saldo_medio_var13_largo_hace2', 'delta_num_venta_var44_1y3', 'var38', 'num_meses_var5_ult3',
         'num_meses_var8_ult3', 'var36', 'num_sal_var16_ult1', 'num_var26_0', 'saldo_medio_var44_ult3', 'ind_var39_0',
         'saldo_medio_var44_ult1', 'num_aport_var17_hace3', 'ind_var10cte_ult1', 'ind_var31_0', 'num_var22_ult1',
         'num_var22_ult3', 'saldo_medio_var12_ult3', 'num_var20', 'imp_compra_var44_hace3', 'imp_sal_var16_ult1',
         'num_var25', 'num_var24', 'saldo_medio_var12_ult1', 'num_var26', 'num_var44_0', 'ind_var6_0',
         'imp_aport_var13_ult1', 'delta_imp_aport_var17_1y3', 'num_meses_var13_largo_ult3',
         'saldo_medio_var13_corto_ult3', 'imp_reemb_var13_ult1', 'saldo_medio_var13_corto_ult1', 'ind_var5_0',
         'num_var29_0', 'num_var12', 'num_var14', 'num_var13', 'num_var32_0', 'num_var17', 'ind_var43_recib_ult1',
         'num_trasp_var11_ult1', 'ind_var13_corto_0', 'num_op_var40_efect_ult3', 'delta_imp_reemb_var13_1y3',
         'num_var40', 'num_var42', 'imp_aport_var33_ult1', 'num_var17_0', 'num_var44', 'ind_var44_0', 'ind_var29_0',
         'num_var20_0', 'saldo_var13_largo', 'imp_aport_var33_hace3', 'var3', 'num_med_var22_ult3', 'num_var13_corto',
         'imp_op_var40_comer_ult3', 'num_op_var40_ult1', 'imp_op_var40_comer_ult1', 'num_op_var40_ult3',
         'ind_var13_largo_0', 'delta_imp_aport_var33_1y3', 'delta_num_aport_var13_1y3', 'saldo_medio_var17_hace2',
         'num_var30', 'num_var32', 'num_var31', 'num_var33', 'num_var31_0', 'num_var35', 'num_meses_var17_ult3',
         'ind_var33_0', 'num_var37', 'num_var39', 'num_var1_0', 'imp_var43_emit_ult1']]


    # Explica lo que se hace en este paso


    categorical_features = []
    numerical_features = ['num_var45_ult1', 'num_op_var39_ult1', 'num_op_var40_comer_ult3', 'num_var45_ult3',
                          'num_aport_var17_ult1', 'delta_imp_reemb_var17_1y3', 'num_compra_var44_hace3',
                          'ind_var37_cte', 'num_op_var39_ult3', 'ind_var40', 'num_var12_0', 'num_op_var40_comer_ult1',
                          'ind_var44', 'ind_var8', 'ind_var24_0', 'ind_var5', 'num_op_var41_hace3', 'ind_var1',
                          'ind_var8_0', 'num_op_var41_efect_ult3', 'num_op_var41_hace2', 'num_op_var39_hace3',
                          'num_op_var39_hace2', 'num_aport_var13_hace3', 'num_aport_var33_hace3',
                          'num_meses_var12_ult3', 'num_op_var41_efect_ult1', 'num_var37_med_ult2',
                          'num_var7_recib_ult1', 'saldo_medio_var33_hace2', 'saldo_medio_var33_hace3',
                          'saldo_medio_var8_hace3', 'saldo_medio_var8_hace2', 'imp_op_var39_ult1', 'num_ent_var16_ult1',
                          'delta_imp_venta_var44_1y3', 'imp_op_var39_efect_ult1', 'ind_var13_0', 'ind_var13_corto',
                          'saldo_medio_var5_ult3', 'imp_op_var39_efect_ult3', 'saldo_medio_var5_ult1',
                          'num_op_var40_efect_ult1', 'num_var8_0', 'imp_op_var39_comer_ult1', 'num_var13_largo_0',
                          'imp_op_var39_comer_ult3', 'num_var45_hace3', 'imp_aport_var13_hace3', 'num_var43_emit_ult1',
                          'num_var45_hace2', 'num_var13_corto_0', 'num_var8', 'num_var4', 'num_var5', 'num_var1',
                          'ind_var12_0', 'num_op_var40_hace2', 'num_var33_0', 'ind_var9_cte_ult1', 'imp_op_var40_ult1',
                          'num_meses_var39_vig_ult3', 'num_var14_0', 'ind_var10_ult1', 'num_var37_0', 'num_var13_largo',
                          'delta_imp_aport_var13_1y3', 'saldo_medio_var12_hace3', 'ind_var26_0',
                          'saldo_medio_var12_hace2', 'num_var40_0', 'ind_var41_0', 'ind_var14', 'ind_var12',
                          'ind_var13', 'ind_var19', 'ind_var26_cte', 'ind_var17', 'ind_var1_0', 'num_var25_0',
                          'ind_var43_emit_ult1', 'num_var22_hace2', 'num_var22_hace3', 'saldo_var13', 'saldo_var12',
                          'num_var6_0', 'saldo_var14', 'saldo_var17', 'imp_op_var41_efect_ult3', 'ind_var32_cte',
                          'imp_op_var41_efect_ult1', 'ind_var30_0', 'ind_var25', 'ind_var26', 'imp_trans_var37_ult1',
                          'num_meses_var33_ult3', 'ind_var24', 'imp_var7_recib_ult1', 'imp_ent_var16_ult1',
                          'imp_aport_var17_hace3', 'num_med_var45_ult3', 'num_var13_0', 'imp_op_var41_comer_ult1',
                          'imp_op_var41_comer_ult3', 'saldo_var20', 'imp_aport_var17_ult1', 'ind_var20', 'ind_var25_0',
                          'saldo_var24', 'saldo_var26', 'saldo_var25', 'num_op_var41_comer_ult3',
                          'num_op_var41_comer_ult1', 'ind_var40_0', 'ind_var37', 'ind_var39', 'ind_var25_cte',
                          'num_var24_0', 'delta_imp_compra_var44_1y3', 'num_aport_var13_ult1', 'ind_var32',
                          'num_reemb_var13_ult1', 'saldo_medio_var33_ult3', 'ind_var33', 'num_venta_var44_ult1',
                          'ind_var30', 'saldo_var31', 'ind_var31', 'saldo_var30', 'saldo_var33', 'saldo_var32',
                          'ind_var14_0', 'saldo_medio_var33_ult1', 'num_var5_0', 'saldo_var37', 'ind_var37_0',
                          'ind_var13_largo', 'saldo_var13_corto', 'num_meses_var44_ult3', 'num_var39_0',
                          'num_var43_recib_ult1', 'var21', 'saldo_var40', 'saldo_medio_var17_ult1', 'saldo_var42',
                          'saldo_medio_var17_ult3', 'saldo_var44', 'num_var42_0', 'delta_num_reemb_var13_1y3',
                          'saldo_medio_var13_largo_ult1', 'num_op_var39_comer_ult3', 'num_op_var39_comer_ult1',
                          'ind_var20_0', 'num_op_var41_ult1', 'num_reemb_var17_ult1', 'saldo_medio_var13_largo_ult3',
                          'num_compra_var44_ult1', 'num_op_var41_ult3', 'num_meses_var29_ult3', 'imp_op_var41_ult1',
                          'ind_var9_ult1', 'delta_num_reemb_var17_1y3', 'var15', 'imp_compra_var44_ult1',
                          'imp_op_var40_efect_ult3', 'imp_op_var40_efect_ult1', 'num_var30_0', 'saldo_var5',
                          'saldo_var8', 'delta_num_aport_var17_1y3', 'saldo_medio_var8_ult3', 'saldo_var1',
                          'ind_var17_0', 'num_aport_var33_ult1', 'saldo_medio_var13_corto_hace2', 'ind_var32_0',
                          'imp_venta_var44_ult1', 'saldo_medio_var5_hace2', 'saldo_medio_var13_corto_hace3',
                          'saldo_medio_var5_hace3', 'delta_num_compra_var44_1y3', 'saldo_medio_var44_hace3',
                          'ind_var7_recib_ult1', 'saldo_medio_var44_hace2', 'saldo_medio_var8_ult1',
                          'delta_num_aport_var33_1y3', 'num_var41_0', 'num_op_var39_efect_ult1',
                          'num_op_var39_efect_ult3', 'saldo_medio_var13_largo_hace3', 'num_meses_var13_corto_ult3',
                          'saldo_medio_var13_largo_hace2', 'delta_num_venta_var44_1y3', 'var38', 'num_meses_var5_ult3',
                          'num_meses_var8_ult3', 'var36', 'num_sal_var16_ult1', 'num_var26_0', 'saldo_medio_var44_ult3',
                          'ind_var39_0', 'saldo_medio_var44_ult1', 'num_aport_var17_hace3', 'ind_var10cte_ult1',
                          'ind_var31_0', 'num_var22_ult1', 'num_var22_ult3', 'saldo_medio_var12_ult3', 'num_var20',
                          'imp_compra_var44_hace3', 'imp_sal_var16_ult1', 'num_var25', 'num_var24',
                          'saldo_medio_var12_ult1', 'num_var26', 'num_var44_0', 'ind_var6_0', 'imp_aport_var13_ult1',
                          'delta_imp_aport_var17_1y3', 'num_meses_var13_largo_ult3', 'saldo_medio_var13_corto_ult3',
                          'imp_reemb_var13_ult1', 'saldo_medio_var13_corto_ult1', 'ind_var5_0', 'num_var29_0',
                          'num_var12', 'num_var14', 'num_var13', 'num_var32_0', 'num_var17', 'ind_var43_recib_ult1',
                          'num_trasp_var11_ult1', 'ind_var13_corto_0', 'num_op_var40_efect_ult3',
                          'delta_imp_reemb_var13_1y3', 'num_var40', 'num_var42', 'imp_aport_var33_ult1', 'num_var17_0',
                          'num_var44', 'ind_var44_0', 'ind_var29_0', 'num_var20_0', 'saldo_var13_largo',
                          'imp_aport_var33_hace3', 'var3', 'num_med_var22_ult3', 'num_var13_corto',
                          'imp_op_var40_comer_ult3', 'num_op_var40_ult1', 'imp_op_var40_comer_ult1',
                          'num_op_var40_ult3', 'ind_var13_largo_0', 'delta_imp_aport_var33_1y3',
                          'delta_num_aport_var13_1y3', 'saldo_medio_var17_hace2', 'num_var30', 'num_var32', 'num_var31',
                          'num_var33', 'num_var31_0', 'num_var35', 'num_meses_var17_ult3', 'ind_var33_0', 'num_var37',
                          'num_var39', 'num_var1_0', 'imp_var43_emit_ult1']
    text_features = []
    for feature in categorical_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in text_features:
        ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
            ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
        else:
            ml_dataset[feature] = ml_dataset[feature].astype('double')



    target_map = {'0': 0, '1': 1}
    ml_dataset['__target__'] = ml_dataset['TARGET'].map(str).map(target_map)
    del ml_dataset['TARGET']

    # Remove rows for which the target is unknown.
    ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]
    print(f)
    print(ml_dataset.head(5))


    train, test = train_test_split(ml_dataset,test_size=0.2,random_state=42,stratify=ml_dataset[['__target__']])
    print(train.head(5))
    print(train['__target__'].value_counts())
    print(test['__target__'].value_counts())

    drop_rows_when_missing = []
    impute_when_missing = [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var45_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_reemb_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_compra_var44_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var37_cte', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var40', 'impute_with': 'MEAN'},
                           {'feature': 'num_var12_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var44', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var8', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var24_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var5', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var8_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var13_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var33_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var12_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var37_med_ult2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var7_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_ent_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_venta_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_corto', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var8_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_largo_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var45_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var13_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var43_emit_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var45_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_corto_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var8', 'impute_with': 'MEAN'},
                           {'feature': 'num_var4', 'impute_with': 'MEAN'},
                           {'feature': 'num_var5', 'impute_with': 'MEAN'},
                           {'feature': 'num_var1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var12_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var33_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var9_cte_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var39_vig_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var14_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var10_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var37_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_largo', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_aport_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var26_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var40_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var41_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var14', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var12', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var19', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var26_cte', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var17', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var1_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var25_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var43_emit_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var13', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var12', 'impute_with': 'MEAN'},
                           {'feature': 'num_var6_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var14', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var17', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var32_cte', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var30_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var25', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var26', 'impute_with': 'MEAN'},
                           {'feature': 'imp_trans_var37_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var33_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var24', 'impute_with': 'MEAN'},
                           {'feature': 'imp_var7_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_ent_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var17_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_med_var45_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var20', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var20', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var25_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var24', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var26', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var25', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var40_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var37', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var39', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var25_cte', 'impute_with': 'MEAN'},
                           {'feature': 'num_var24_0', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_compra_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var32', 'impute_with': 'MEAN'},
                           {'feature': 'num_reemb_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var33', 'impute_with': 'MEAN'},
                           {'feature': 'num_venta_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var30', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var31', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var31', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var30', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var33', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var32', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var14_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var5_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var37', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var37_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_largo', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var13_corto', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var44_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var39_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var43_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'var21', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var40', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var42', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var17_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var44', 'impute_with': 'MEAN'},
                           {'feature': 'num_var42_0', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_reemb_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var20_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_reemb_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_compra_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var29_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var9_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_reemb_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'var15', 'impute_with': 'MEAN'},
                           {'feature': 'imp_compra_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var30_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var5', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var8', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_aport_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var17_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var33_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var32_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_venta_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_compra_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var7_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_aport_var33_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var41_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var13_corto_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_venta_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'var38', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var5_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var8_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'var36', 'impute_with': 'MEAN'},
                           {'feature': 'num_sal_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var26_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var39_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var17_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var10cte_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var31_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var20', 'impute_with': 'MEAN'},
                           {'feature': 'imp_compra_var44_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_sal_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var25', 'impute_with': 'MEAN'},
                           {'feature': 'num_var24', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var26', 'impute_with': 'MEAN'},
                           {'feature': 'num_var44_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var6_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_aport_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var13_largo_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_reemb_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var5_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var29_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var12', 'impute_with': 'MEAN'},
                           {'feature': 'num_var14', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13', 'impute_with': 'MEAN'},
                           {'feature': 'num_var32_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var17', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var43_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_trasp_var11_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_corto_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_reemb_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var40', 'impute_with': 'MEAN'},
                           {'feature': 'num_var42', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var33_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var17_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var44', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var44_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var29_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var20_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var13_largo', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var33_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'var3', 'impute_with': 'MEAN'},
                           {'feature': 'num_med_var22_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_corto', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_largo_0', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_aport_var33_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_aport_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var17_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var30', 'impute_with': 'MEAN'},
                           {'feature': 'num_var32', 'impute_with': 'MEAN'},
                           {'feature': 'num_var31', 'impute_with': 'MEAN'},
                           {'feature': 'num_var33', 'impute_with': 'MEAN'},
                           {'feature': 'num_var31_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var35', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var17_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var33_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var37', 'impute_with': 'MEAN'},
                           {'feature': 'num_var39', 'impute_with': 'MEAN'},
                           {'feature': 'num_var1_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_var43_emit_ult1', 'impute_with': 'MEAN'}]

    # Explica lo que se hace en este paso
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Explica lo que se hace en este paso
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))



    rescale_features = {'num_var45_ult1': 'AVGSTD', 'num_op_var39_ult1': 'AVGSTD', 'num_op_var40_comer_ult3': 'AVGSTD',
                        'num_var45_ult3': 'AVGSTD', 'num_aport_var17_ult1': 'AVGSTD',
                        'delta_imp_reemb_var17_1y3': 'AVGSTD', 'num_compra_var44_hace3': 'AVGSTD',
                        'ind_var37_cte': 'AVGSTD', 'num_op_var39_ult3': 'AVGSTD', 'ind_var40': 'AVGSTD',
                        'num_var12_0': 'AVGSTD', 'num_op_var40_comer_ult1': 'AVGSTD', 'ind_var44': 'AVGSTD',
                        'ind_var8': 'AVGSTD', 'ind_var24_0': 'AVGSTD', 'ind_var5': 'AVGSTD',
                        'num_op_var41_hace3': 'AVGSTD', 'ind_var1': 'AVGSTD', 'ind_var8_0': 'AVGSTD',
                        'num_op_var41_efect_ult3': 'AVGSTD', 'num_op_var41_hace2': 'AVGSTD',
                        'num_op_var39_hace3': 'AVGSTD', 'num_op_var39_hace2': 'AVGSTD',
                        'num_aport_var13_hace3': 'AVGSTD', 'num_aport_var33_hace3': 'AVGSTD',
                        'num_meses_var12_ult3': 'AVGSTD', 'num_op_var41_efect_ult1': 'AVGSTD',
                        'num_var37_med_ult2': 'AVGSTD', 'num_var7_recib_ult1': 'AVGSTD',
                        'saldo_medio_var33_hace2': 'AVGSTD', 'saldo_medio_var33_hace3': 'AVGSTD',
                        'saldo_medio_var8_hace3': 'AVGSTD', 'saldo_medio_var8_hace2': 'AVGSTD',
                        'imp_op_var39_ult1': 'AVGSTD', 'num_ent_var16_ult1': 'AVGSTD',
                        'delta_imp_venta_var44_1y3': 'AVGSTD', 'imp_op_var39_efect_ult1': 'AVGSTD',
                        'ind_var13_0': 'AVGSTD', 'ind_var13_corto': 'AVGSTD', 'saldo_medio_var5_ult3': 'AVGSTD',
                        'imp_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var5_ult1': 'AVGSTD',
                        'num_op_var40_efect_ult1': 'AVGSTD', 'num_var8_0': 'AVGSTD',
                        'imp_op_var39_comer_ult1': 'AVGSTD', 'num_var13_largo_0': 'AVGSTD',
                        'imp_op_var39_comer_ult3': 'AVGSTD', 'num_var45_hace3': 'AVGSTD',
                        'imp_aport_var13_hace3': 'AVGSTD', 'num_var43_emit_ult1': 'AVGSTD', 'num_var45_hace2': 'AVGSTD',
                        'num_var13_corto_0': 'AVGSTD', 'num_var8': 'AVGSTD', 'num_var4': 'AVGSTD', 'num_var5': 'AVGSTD',
                        'num_var1': 'AVGSTD', 'ind_var12_0': 'AVGSTD', 'num_op_var40_hace2': 'AVGSTD',
                        'num_var33_0': 'AVGSTD', 'ind_var9_cte_ult1': 'AVGSTD', 'imp_op_var40_ult1': 'AVGSTD',
                        'num_meses_var39_vig_ult3': 'AVGSTD', 'num_var14_0': 'AVGSTD', 'ind_var10_ult1': 'AVGSTD',
                        'num_var37_0': 'AVGSTD', 'num_var13_largo': 'AVGSTD', 'delta_imp_aport_var13_1y3': 'AVGSTD',
                        'saldo_medio_var12_hace3': 'AVGSTD', 'ind_var26_0': 'AVGSTD',
                        'saldo_medio_var12_hace2': 'AVGSTD', 'num_var40_0': 'AVGSTD', 'ind_var41_0': 'AVGSTD',
                        'ind_var14': 'AVGSTD', 'ind_var12': 'AVGSTD', 'ind_var13': 'AVGSTD', 'ind_var19': 'AVGSTD',
                        'ind_var26_cte': 'AVGSTD', 'ind_var17': 'AVGSTD', 'ind_var1_0': 'AVGSTD',
                        'num_var25_0': 'AVGSTD', 'ind_var43_emit_ult1': 'AVGSTD', 'num_var22_hace2': 'AVGSTD',
                        'num_var22_hace3': 'AVGSTD', 'saldo_var13': 'AVGSTD', 'saldo_var12': 'AVGSTD',
                        'num_var6_0': 'AVGSTD', 'saldo_var14': 'AVGSTD', 'saldo_var17': 'AVGSTD',
                        'imp_op_var41_efect_ult3': 'AVGSTD', 'ind_var32_cte': 'AVGSTD',
                        'imp_op_var41_efect_ult1': 'AVGSTD', 'ind_var30_0': 'AVGSTD', 'ind_var25': 'AVGSTD',
                        'ind_var26': 'AVGSTD', 'imp_trans_var37_ult1': 'AVGSTD', 'num_meses_var33_ult3': 'AVGSTD',
                        'ind_var24': 'AVGSTD', 'imp_var7_recib_ult1': 'AVGSTD', 'imp_ent_var16_ult1': 'AVGSTD',
                        'imp_aport_var17_hace3': 'AVGSTD', 'num_med_var45_ult3': 'AVGSTD', 'num_var13_0': 'AVGSTD',
                        'imp_op_var41_comer_ult1': 'AVGSTD', 'imp_op_var41_comer_ult3': 'AVGSTD',
                        'saldo_var20': 'AVGSTD', 'imp_aport_var17_ult1': 'AVGSTD', 'ind_var20': 'AVGSTD',
                        'ind_var25_0': 'AVGSTD', 'saldo_var24': 'AVGSTD', 'saldo_var26': 'AVGSTD',
                        'saldo_var25': 'AVGSTD', 'num_op_var41_comer_ult3': 'AVGSTD',
                        'num_op_var41_comer_ult1': 'AVGSTD', 'ind_var40_0': 'AVGSTD', 'ind_var37': 'AVGSTD',
                        'ind_var39': 'AVGSTD', 'ind_var25_cte': 'AVGSTD', 'num_var24_0': 'AVGSTD',
                        'delta_imp_compra_var44_1y3': 'AVGSTD', 'num_aport_var13_ult1': 'AVGSTD', 'ind_var32': 'AVGSTD',
                        'num_reemb_var13_ult1': 'AVGSTD', 'saldo_medio_var33_ult3': 'AVGSTD', 'ind_var33': 'AVGSTD',
                        'num_venta_var44_ult1': 'AVGSTD', 'ind_var30': 'AVGSTD', 'saldo_var31': 'AVGSTD',
                        'ind_var31': 'AVGSTD', 'saldo_var30': 'AVGSTD', 'saldo_var33': 'AVGSTD',
                        'saldo_var32': 'AVGSTD', 'ind_var14_0': 'AVGSTD', 'saldo_medio_var33_ult1': 'AVGSTD',
                        'num_var5_0': 'AVGSTD', 'saldo_var37': 'AVGSTD', 'ind_var37_0': 'AVGSTD',
                        'ind_var13_largo': 'AVGSTD', 'saldo_var13_corto': 'AVGSTD', 'num_meses_var44_ult3': 'AVGSTD',
                        'num_var39_0': 'AVGSTD', 'num_var43_recib_ult1': 'AVGSTD', 'var21': 'AVGSTD',
                        'saldo_var40': 'AVGSTD', 'saldo_medio_var17_ult1': 'AVGSTD', 'saldo_var42': 'AVGSTD',
                        'saldo_medio_var17_ult3': 'AVGSTD', 'saldo_var44': 'AVGSTD', 'num_var42_0': 'AVGSTD',
                        'delta_num_reemb_var13_1y3': 'AVGSTD', 'saldo_medio_var13_largo_ult1': 'AVGSTD',
                        'num_op_var39_comer_ult3': 'AVGSTD', 'num_op_var39_comer_ult1': 'AVGSTD',
                        'ind_var20_0': 'AVGSTD', 'num_op_var41_ult1': 'AVGSTD', 'num_reemb_var17_ult1': 'AVGSTD',
                        'saldo_medio_var13_largo_ult3': 'AVGSTD', 'num_compra_var44_ult1': 'AVGSTD',
                        'num_op_var41_ult3': 'AVGSTD', 'num_meses_var29_ult3': 'AVGSTD', 'imp_op_var41_ult1': 'AVGSTD',
                        'ind_var9_ult1': 'AVGSTD', 'delta_num_reemb_var17_1y3': 'AVGSTD', 'var15': 'AVGSTD',
                        'imp_compra_var44_ult1': 'AVGSTD', 'imp_op_var40_efect_ult3': 'AVGSTD',
                        'imp_op_var40_efect_ult1': 'AVGSTD', 'num_var30_0': 'AVGSTD', 'saldo_var5': 'AVGSTD',
                        'saldo_var8': 'AVGSTD', 'delta_num_aport_var17_1y3': 'AVGSTD',
                        'saldo_medio_var8_ult3': 'AVGSTD', 'saldo_var1': 'AVGSTD', 'ind_var17_0': 'AVGSTD',
                        'num_aport_var33_ult1': 'AVGSTD', 'saldo_medio_var13_corto_hace2': 'AVGSTD',
                        'ind_var32_0': 'AVGSTD', 'imp_venta_var44_ult1': 'AVGSTD', 'saldo_medio_var5_hace2': 'AVGSTD',
                        'saldo_medio_var13_corto_hace3': 'AVGSTD', 'saldo_medio_var5_hace3': 'AVGSTD',
                        'delta_num_compra_var44_1y3': 'AVGSTD', 'saldo_medio_var44_hace3': 'AVGSTD',
                        'ind_var7_recib_ult1': 'AVGSTD', 'saldo_medio_var44_hace2': 'AVGSTD',
                        'saldo_medio_var8_ult1': 'AVGSTD', 'delta_num_aport_var33_1y3': 'AVGSTD',
                        'num_var41_0': 'AVGSTD', 'num_op_var39_efect_ult1': 'AVGSTD',
                        'num_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace3': 'AVGSTD',
                        'num_meses_var13_corto_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace2': 'AVGSTD',
                        'delta_num_venta_var44_1y3': 'AVGSTD', 'var38': 'AVGSTD', 'num_meses_var5_ult3': 'AVGSTD',
                        'num_meses_var8_ult3': 'AVGSTD', 'var36': 'AVGSTD', 'num_sal_var16_ult1': 'AVGSTD',
                        'num_var26_0': 'AVGSTD', 'saldo_medio_var44_ult3': 'AVGSTD', 'ind_var39_0': 'AVGSTD',
                        'saldo_medio_var44_ult1': 'AVGSTD', 'num_aport_var17_hace3': 'AVGSTD',
                        'ind_var10cte_ult1': 'AVGSTD', 'ind_var31_0': 'AVGSTD', 'num_var22_ult1': 'AVGSTD',
                        'num_var22_ult3': 'AVGSTD', 'saldo_medio_var12_ult3': 'AVGSTD', 'num_var20': 'AVGSTD',
                        'imp_compra_var44_hace3': 'AVGSTD', 'imp_sal_var16_ult1': 'AVGSTD', 'num_var25': 'AVGSTD',
                        'num_var24': 'AVGSTD', 'saldo_medio_var12_ult1': 'AVGSTD', 'num_var26': 'AVGSTD',
                        'num_var44_0': 'AVGSTD', 'ind_var6_0': 'AVGSTD', 'imp_aport_var13_ult1': 'AVGSTD',
                        'delta_imp_aport_var17_1y3': 'AVGSTD', 'num_meses_var13_largo_ult3': 'AVGSTD',
                        'saldo_medio_var13_corto_ult3': 'AVGSTD', 'imp_reemb_var13_ult1': 'AVGSTD',
                        'saldo_medio_var13_corto_ult1': 'AVGSTD', 'ind_var5_0': 'AVGSTD', 'num_var29_0': 'AVGSTD',
                        'num_var12': 'AVGSTD', 'num_var14': 'AVGSTD', 'num_var13': 'AVGSTD', 'num_var32_0': 'AVGSTD',
                        'num_var17': 'AVGSTD', 'ind_var43_recib_ult1': 'AVGSTD', 'num_trasp_var11_ult1': 'AVGSTD',
                        'ind_var13_corto_0': 'AVGSTD', 'num_op_var40_efect_ult3': 'AVGSTD',
                        'delta_imp_reemb_var13_1y3': 'AVGSTD', 'num_var40': 'AVGSTD', 'num_var42': 'AVGSTD',
                        'imp_aport_var33_ult1': 'AVGSTD', 'num_var17_0': 'AVGSTD', 'num_var44': 'AVGSTD',
                        'ind_var44_0': 'AVGSTD', 'ind_var29_0': 'AVGSTD', 'num_var20_0': 'AVGSTD',
                        'saldo_var13_largo': 'AVGSTD', 'imp_aport_var33_hace3': 'AVGSTD', 'var3': 'AVGSTD',
                        'num_med_var22_ult3': 'AVGSTD', 'num_var13_corto': 'AVGSTD',
                        'imp_op_var40_comer_ult3': 'AVGSTD', 'num_op_var40_ult1': 'AVGSTD',
                        'imp_op_var40_comer_ult1': 'AVGSTD', 'num_op_var40_ult3': 'AVGSTD',
                        'ind_var13_largo_0': 'AVGSTD', 'delta_imp_aport_var33_1y3': 'AVGSTD',
                        'delta_num_aport_var13_1y3': 'AVGSTD', 'saldo_medio_var17_hace2': 'AVGSTD',
                        'num_var30': 'AVGSTD', 'num_var32': 'AVGSTD', 'num_var31': 'AVGSTD', 'num_var33': 'AVGSTD',
                        'num_var31_0': 'AVGSTD', 'num_var35': 'AVGSTD', 'num_meses_var17_ult3': 'AVGSTD',
                        'ind_var33_0': 'AVGSTD', 'num_var37': 'AVGSTD', 'num_var39': 'AVGSTD', 'num_var1_0': 'AVGSTD',
                        'imp_var43_emit_ult1': 'AVGSTD'}
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    drop_rows_when_missing = []
    impute_when_missing = [{'feature': 'num_var45_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var45_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_reemb_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_compra_var44_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var37_cte', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var40', 'impute_with': 'MEAN'},
                           {'feature': 'num_var12_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var44', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var8', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var24_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var5', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var8_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var13_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var33_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var12_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var37_med_ult2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var7_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_ent_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_venta_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_corto', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var8_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_largo_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var39_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var45_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var13_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var43_emit_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var45_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_corto_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var8', 'impute_with': 'MEAN'},
                           {'feature': 'num_var4', 'impute_with': 'MEAN'},
                           {'feature': 'num_var5', 'impute_with': 'MEAN'},
                           {'feature': 'num_var1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var12_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var33_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var9_cte_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var39_vig_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var14_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var10_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var37_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_largo', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_aport_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var26_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var40_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var41_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var14', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var12', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var19', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var26_cte', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var17', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var1_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var25_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var43_emit_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var13', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var12', 'impute_with': 'MEAN'},
                           {'feature': 'num_var6_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var14', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var17', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var32_cte', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var30_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var25', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var26', 'impute_with': 'MEAN'},
                           {'feature': 'imp_trans_var37_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var33_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var24', 'impute_with': 'MEAN'},
                           {'feature': 'imp_var7_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_ent_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var17_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_med_var45_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var20', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var20', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var25_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var24', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var26', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var25', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var40_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var37', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var39', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var25_cte', 'impute_with': 'MEAN'},
                           {'feature': 'num_var24_0', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_compra_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var32', 'impute_with': 'MEAN'},
                           {'feature': 'num_reemb_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var33', 'impute_with': 'MEAN'},
                           {'feature': 'num_venta_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var30', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var31', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var31', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var30', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var33', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var32', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var14_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var33_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var5_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var37', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var37_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_largo', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var13_corto', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var44_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var39_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var43_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'var21', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var40', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var42', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var17_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var44', 'impute_with': 'MEAN'},
                           {'feature': 'num_var42_0', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_reemb_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var20_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_reemb_var17_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_compra_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var41_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var29_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var41_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var9_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_reemb_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'var15', 'impute_with': 'MEAN'},
                           {'feature': 'imp_compra_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var30_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var5', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var8', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_aport_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var17_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var33_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var32_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_venta_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var5_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_compra_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var7_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var8_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_aport_var33_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var41_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_efect_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var39_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var13_corto_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_largo_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_venta_var44_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'var38', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var5_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var8_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'var36', 'impute_with': 'MEAN'},
                           {'feature': 'num_sal_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var26_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var39_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var44_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_aport_var17_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var10cte_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var31_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var22_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var20', 'impute_with': 'MEAN'},
                           {'feature': 'imp_compra_var44_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_sal_var16_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var25', 'impute_with': 'MEAN'},
                           {'feature': 'num_var24', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var12_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var26', 'impute_with': 'MEAN'},
                           {'feature': 'num_var44_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var6_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_aport_var17_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var13_largo_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'imp_reemb_var13_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var13_corto_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var5_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var29_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var12', 'impute_with': 'MEAN'},
                           {'feature': 'num_var14', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13', 'impute_with': 'MEAN'},
                           {'feature': 'num_var32_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var17', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var43_recib_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_trasp_var11_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_corto_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_efect_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_reemb_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var40', 'impute_with': 'MEAN'},
                           {'feature': 'num_var42', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var33_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_var17_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var44', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var44_0', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var29_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var20_0', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_var13_largo', 'impute_with': 'MEAN'},
                           {'feature': 'imp_aport_var33_hace3', 'impute_with': 'MEAN'},
                           {'feature': 'var3', 'impute_with': 'MEAN'},
                           {'feature': 'num_med_var22_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_var13_corto', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_comer_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'imp_op_var40_comer_ult1', 'impute_with': 'MEAN'},
                           {'feature': 'num_op_var40_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var13_largo_0', 'impute_with': 'MEAN'},
                           {'feature': 'delta_imp_aport_var33_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'delta_num_aport_var13_1y3', 'impute_with': 'MEAN'},
                           {'feature': 'saldo_medio_var17_hace2', 'impute_with': 'MEAN'},
                           {'feature': 'num_var30', 'impute_with': 'MEAN'},
                           {'feature': 'num_var32', 'impute_with': 'MEAN'},
                           {'feature': 'num_var31', 'impute_with': 'MEAN'},
                           {'feature': 'num_var33', 'impute_with': 'MEAN'},
                           {'feature': 'num_var31_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var35', 'impute_with': 'MEAN'},
                           {'feature': 'num_meses_var17_ult3', 'impute_with': 'MEAN'},
                           {'feature': 'ind_var33_0', 'impute_with': 'MEAN'},
                           {'feature': 'num_var37', 'impute_with': 'MEAN'},
                           {'feature': 'num_var39', 'impute_with': 'MEAN'},
                           {'feature': 'num_var1_0', 'impute_with': 'MEAN'},
                           {'feature': 'imp_var43_emit_ult1', 'impute_with': 'MEAN'}]

    # Features for which we drop rows with missing values"
    for feature in drop_rows_when_missing:
        train = train[train[feature].notnull()]
        test = test[test[feature].notnull()]
        print('Dropped missing records in %s' % feature)

    # Explica lo que se hace en este paso
    for feature in impute_when_missing:
        if feature['impute_with'] == 'MEAN':
            v = train[feature['feature']].mean()
        elif feature['impute_with'] == 'MEDIAN':
            v = train[feature['feature']].median()
        elif feature['impute_with'] == 'CREATE_CATEGORY':
            v = 'NULL_CATEGORY'
        elif feature['impute_with'] == 'MODE':
            v = train[feature['feature']].value_counts().index[0]
        elif feature['impute_with'] == 'CONSTANT':
            v = feature['value']
        train[feature['feature']] = train[feature['feature']].fillna(v)
        test[feature['feature']] = test[feature['feature']].fillna(v)
        print('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))



    rescale_features = {'num_var45_ult1': 'AVGSTD', 'num_op_var39_ult1': 'AVGSTD', 'num_op_var40_comer_ult3': 'AVGSTD',
                        'num_var45_ult3': 'AVGSTD', 'num_aport_var17_ult1': 'AVGSTD',
                        'delta_imp_reemb_var17_1y3': 'AVGSTD', 'num_compra_var44_hace3': 'AVGSTD',
                        'ind_var37_cte': 'AVGSTD', 'num_op_var39_ult3': 'AVGSTD', 'ind_var40': 'AVGSTD',
                        'num_var12_0': 'AVGSTD', 'num_op_var40_comer_ult1': 'AVGSTD', 'ind_var44': 'AVGSTD',
                        'ind_var8': 'AVGSTD', 'ind_var24_0': 'AVGSTD', 'ind_var5': 'AVGSTD',
                        'num_op_var41_hace3': 'AVGSTD', 'ind_var1': 'AVGSTD', 'ind_var8_0': 'AVGSTD',
                        'num_op_var41_efect_ult3': 'AVGSTD', 'num_op_var41_hace2': 'AVGSTD',
                        'num_op_var39_hace3': 'AVGSTD', 'num_op_var39_hace2': 'AVGSTD',
                        'num_aport_var13_hace3': 'AVGSTD', 'num_aport_var33_hace3': 'AVGSTD',
                        'num_meses_var12_ult3': 'AVGSTD', 'num_op_var41_efect_ult1': 'AVGSTD',
                        'num_var37_med_ult2': 'AVGSTD', 'num_var7_recib_ult1': 'AVGSTD',
                        'saldo_medio_var33_hace2': 'AVGSTD', 'saldo_medio_var33_hace3': 'AVGSTD',
                        'saldo_medio_var8_hace3': 'AVGSTD', 'saldo_medio_var8_hace2': 'AVGSTD',
                        'imp_op_var39_ult1': 'AVGSTD', 'num_ent_var16_ult1': 'AVGSTD',
                        'delta_imp_venta_var44_1y3': 'AVGSTD', 'imp_op_var39_efect_ult1': 'AVGSTD',
                        'ind_var13_0': 'AVGSTD', 'ind_var13_corto': 'AVGSTD', 'saldo_medio_var5_ult3': 'AVGSTD',
                        'imp_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var5_ult1': 'AVGSTD',
                        'num_op_var40_efect_ult1': 'AVGSTD', 'num_var8_0': 'AVGSTD',
                        'imp_op_var39_comer_ult1': 'AVGSTD', 'num_var13_largo_0': 'AVGSTD',
                        'imp_op_var39_comer_ult3': 'AVGSTD', 'num_var45_hace3': 'AVGSTD',
                        'imp_aport_var13_hace3': 'AVGSTD', 'num_var43_emit_ult1': 'AVGSTD', 'num_var45_hace2': 'AVGSTD',
                        'num_var13_corto_0': 'AVGSTD', 'num_var8': 'AVGSTD', 'num_var4': 'AVGSTD', 'num_var5': 'AVGSTD',
                        'num_var1': 'AVGSTD', 'ind_var12_0': 'AVGSTD', 'num_op_var40_hace2': 'AVGSTD',
                        'num_var33_0': 'AVGSTD', 'ind_var9_cte_ult1': 'AVGSTD', 'imp_op_var40_ult1': 'AVGSTD',
                        'num_meses_var39_vig_ult3': 'AVGSTD', 'num_var14_0': 'AVGSTD', 'ind_var10_ult1': 'AVGSTD',
                        'num_var37_0': 'AVGSTD', 'num_var13_largo': 'AVGSTD', 'delta_imp_aport_var13_1y3': 'AVGSTD',
                        'saldo_medio_var12_hace3': 'AVGSTD', 'ind_var26_0': 'AVGSTD',
                        'saldo_medio_var12_hace2': 'AVGSTD', 'num_var40_0': 'AVGSTD', 'ind_var41_0': 'AVGSTD',
                        'ind_var14': 'AVGSTD', 'ind_var12': 'AVGSTD', 'ind_var13': 'AVGSTD', 'ind_var19': 'AVGSTD',
                        'ind_var26_cte': 'AVGSTD', 'ind_var17': 'AVGSTD', 'ind_var1_0': 'AVGSTD',
                        'num_var25_0': 'AVGSTD', 'ind_var43_emit_ult1': 'AVGSTD', 'num_var22_hace2': 'AVGSTD',
                        'num_var22_hace3': 'AVGSTD', 'saldo_var13': 'AVGSTD', 'saldo_var12': 'AVGSTD',
                        'num_var6_0': 'AVGSTD', 'saldo_var14': 'AVGSTD', 'saldo_var17': 'AVGSTD',
                        'imp_op_var41_efect_ult3': 'AVGSTD', 'ind_var32_cte': 'AVGSTD',
                        'imp_op_var41_efect_ult1': 'AVGSTD', 'ind_var30_0': 'AVGSTD', 'ind_var25': 'AVGSTD',
                        'ind_var26': 'AVGSTD', 'imp_trans_var37_ult1': 'AVGSTD', 'num_meses_var33_ult3': 'AVGSTD',
                        'ind_var24': 'AVGSTD', 'imp_var7_recib_ult1': 'AVGSTD', 'imp_ent_var16_ult1': 'AVGSTD',
                        'imp_aport_var17_hace3': 'AVGSTD', 'num_med_var45_ult3': 'AVGSTD', 'num_var13_0': 'AVGSTD',
                        'imp_op_var41_comer_ult1': 'AVGSTD', 'imp_op_var41_comer_ult3': 'AVGSTD',
                        'saldo_var20': 'AVGSTD', 'imp_aport_var17_ult1': 'AVGSTD', 'ind_var20': 'AVGSTD',
                        'ind_var25_0': 'AVGSTD', 'saldo_var24': 'AVGSTD', 'saldo_var26': 'AVGSTD',
                        'saldo_var25': 'AVGSTD', 'num_op_var41_comer_ult3': 'AVGSTD',
                        'num_op_var41_comer_ult1': 'AVGSTD', 'ind_var40_0': 'AVGSTD', 'ind_var37': 'AVGSTD',
                        'ind_var39': 'AVGSTD', 'ind_var25_cte': 'AVGSTD', 'num_var24_0': 'AVGSTD',
                        'delta_imp_compra_var44_1y3': 'AVGSTD', 'num_aport_var13_ult1': 'AVGSTD', 'ind_var32': 'AVGSTD',
                        'num_reemb_var13_ult1': 'AVGSTD', 'saldo_medio_var33_ult3': 'AVGSTD', 'ind_var33': 'AVGSTD',
                        'num_venta_var44_ult1': 'AVGSTD', 'ind_var30': 'AVGSTD', 'saldo_var31': 'AVGSTD',
                        'ind_var31': 'AVGSTD', 'saldo_var30': 'AVGSTD', 'saldo_var33': 'AVGSTD',
                        'saldo_var32': 'AVGSTD', 'ind_var14_0': 'AVGSTD', 'saldo_medio_var33_ult1': 'AVGSTD',
                        'num_var5_0': 'AVGSTD', 'saldo_var37': 'AVGSTD', 'ind_var37_0': 'AVGSTD',
                        'ind_var13_largo': 'AVGSTD', 'saldo_var13_corto': 'AVGSTD', 'num_meses_var44_ult3': 'AVGSTD',
                        'num_var39_0': 'AVGSTD', 'num_var43_recib_ult1': 'AVGSTD', 'var21': 'AVGSTD',
                        'saldo_var40': 'AVGSTD', 'saldo_medio_var17_ult1': 'AVGSTD', 'saldo_var42': 'AVGSTD',
                        'saldo_medio_var17_ult3': 'AVGSTD', 'saldo_var44': 'AVGSTD', 'num_var42_0': 'AVGSTD',
                        'delta_num_reemb_var13_1y3': 'AVGSTD', 'saldo_medio_var13_largo_ult1': 'AVGSTD',
                        'num_op_var39_comer_ult3': 'AVGSTD', 'num_op_var39_comer_ult1': 'AVGSTD',
                        'ind_var20_0': 'AVGSTD', 'num_op_var41_ult1': 'AVGSTD', 'num_reemb_var17_ult1': 'AVGSTD',
                        'saldo_medio_var13_largo_ult3': 'AVGSTD', 'num_compra_var44_ult1': 'AVGSTD',
                        'num_op_var41_ult3': 'AVGSTD', 'num_meses_var29_ult3': 'AVGSTD', 'imp_op_var41_ult1': 'AVGSTD',
                        'ind_var9_ult1': 'AVGSTD', 'delta_num_reemb_var17_1y3': 'AVGSTD', 'var15': 'AVGSTD',
                        'imp_compra_var44_ult1': 'AVGSTD', 'imp_op_var40_efect_ult3': 'AVGSTD',
                        'imp_op_var40_efect_ult1': 'AVGSTD', 'num_var30_0': 'AVGSTD', 'saldo_var5': 'AVGSTD',
                        'saldo_var8': 'AVGSTD', 'delta_num_aport_var17_1y3': 'AVGSTD',
                        'saldo_medio_var8_ult3': 'AVGSTD', 'saldo_var1': 'AVGSTD', 'ind_var17_0': 'AVGSTD',
                        'num_aport_var33_ult1': 'AVGSTD', 'saldo_medio_var13_corto_hace2': 'AVGSTD',
                        'ind_var32_0': 'AVGSTD', 'imp_venta_var44_ult1': 'AVGSTD', 'saldo_medio_var5_hace2': 'AVGSTD',
                        'saldo_medio_var13_corto_hace3': 'AVGSTD', 'saldo_medio_var5_hace3': 'AVGSTD',
                        'delta_num_compra_var44_1y3': 'AVGSTD', 'saldo_medio_var44_hace3': 'AVGSTD',
                        'ind_var7_recib_ult1': 'AVGSTD', 'saldo_medio_var44_hace2': 'AVGSTD',
                        'saldo_medio_var8_ult1': 'AVGSTD', 'delta_num_aport_var33_1y3': 'AVGSTD',
                        'num_var41_0': 'AVGSTD', 'num_op_var39_efect_ult1': 'AVGSTD',
                        'num_op_var39_efect_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace3': 'AVGSTD',
                        'num_meses_var13_corto_ult3': 'AVGSTD', 'saldo_medio_var13_largo_hace2': 'AVGSTD',
                        'delta_num_venta_var44_1y3': 'AVGSTD', 'var38': 'AVGSTD', 'num_meses_var5_ult3': 'AVGSTD',
                        'num_meses_var8_ult3': 'AVGSTD', 'var36': 'AVGSTD', 'num_sal_var16_ult1': 'AVGSTD',
                        'num_var26_0': 'AVGSTD', 'saldo_medio_var44_ult3': 'AVGSTD', 'ind_var39_0': 'AVGSTD',
                        'saldo_medio_var44_ult1': 'AVGSTD', 'num_aport_var17_hace3': 'AVGSTD',
                        'ind_var10cte_ult1': 'AVGSTD', 'ind_var31_0': 'AVGSTD', 'num_var22_ult1': 'AVGSTD',
                        'num_var22_ult3': 'AVGSTD', 'saldo_medio_var12_ult3': 'AVGSTD', 'num_var20': 'AVGSTD',
                        'imp_compra_var44_hace3': 'AVGSTD', 'imp_sal_var16_ult1': 'AVGSTD', 'num_var25': 'AVGSTD',
                        'num_var24': 'AVGSTD', 'saldo_medio_var12_ult1': 'AVGSTD', 'num_var26': 'AVGSTD',
                        'num_var44_0': 'AVGSTD', 'ind_var6_0': 'AVGSTD', 'imp_aport_var13_ult1': 'AVGSTD',
                        'delta_imp_aport_var17_1y3': 'AVGSTD', 'num_meses_var13_largo_ult3': 'AVGSTD',
                        'saldo_medio_var13_corto_ult3': 'AVGSTD', 'imp_reemb_var13_ult1': 'AVGSTD',
                        'saldo_medio_var13_corto_ult1': 'AVGSTD', 'ind_var5_0': 'AVGSTD', 'num_var29_0': 'AVGSTD',
                        'num_var12': 'AVGSTD', 'num_var14': 'AVGSTD', 'num_var13': 'AVGSTD', 'num_var32_0': 'AVGSTD',
                        'num_var17': 'AVGSTD', 'ind_var43_recib_ult1': 'AVGSTD', 'num_trasp_var11_ult1': 'AVGSTD',
                        'ind_var13_corto_0': 'AVGSTD', 'num_op_var40_efect_ult3': 'AVGSTD',
                        'delta_imp_reemb_var13_1y3': 'AVGSTD', 'num_var40': 'AVGSTD', 'num_var42': 'AVGSTD',
                        'imp_aport_var33_ult1': 'AVGSTD', 'num_var17_0': 'AVGSTD', 'num_var44': 'AVGSTD',
                        'ind_var44_0': 'AVGSTD', 'ind_var29_0': 'AVGSTD', 'num_var20_0': 'AVGSTD',
                        'saldo_var13_largo': 'AVGSTD', 'imp_aport_var33_hace3': 'AVGSTD', 'var3': 'AVGSTD',
                        'num_med_var22_ult3': 'AVGSTD', 'num_var13_corto': 'AVGSTD',
                        'imp_op_var40_comer_ult3': 'AVGSTD', 'num_op_var40_ult1': 'AVGSTD',
                        'imp_op_var40_comer_ult1': 'AVGSTD', 'num_op_var40_ult3': 'AVGSTD',
                        'ind_var13_largo_0': 'AVGSTD', 'delta_imp_aport_var33_1y3': 'AVGSTD',
                        'delta_num_aport_var13_1y3': 'AVGSTD', 'saldo_medio_var17_hace2': 'AVGSTD',
                        'num_var30': 'AVGSTD', 'num_var32': 'AVGSTD', 'num_var31': 'AVGSTD', 'num_var33': 'AVGSTD',
                        'num_var31_0': 'AVGSTD', 'num_var35': 'AVGSTD', 'num_meses_var17_ult3': 'AVGSTD',
                        'ind_var33_0': 'AVGSTD', 'num_var37': 'AVGSTD', 'num_var39': 'AVGSTD', 'num_var1_0': 'AVGSTD',
                        'imp_var43_emit_ult1': 'AVGSTD'}

    # Explica lo que se hace en este paso
    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            _min = train[feature_name].min()
            _max = train[feature_name].max()
            scale = _max - _min
            shift = _min
        else:
            shift = train[feature_name].mean()
            scale = train[feature_name].std()
        if scale == 0.:
            del train[feature_name]
            del test[feature_name]
            print('Feature %s was dropped because it has no variance' % feature_name)
        else:
            print('Rescaled %s' % feature_name)
            train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
            test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

    print("TRAIN: ")
    print(train.head(5))
    print("DEV: ")
    print(test.head(5))

    trainX = train.drop('__target__', axis=1)
    #trainY = train['__target__']

    testX = test.drop('__target__', axis=1)
    #testY = test['__target__']

    trainY = np.array(train['__target__'])
    testY = np.array(test['__target__'])

    # Explica lo que se hace en este paso
    undersample = RandomUnderSampler(sampling_strategy=0.5)#la mayoria va a estar representada el doble de veces

    trainXUnder,trainYUnder = undersample.fit_resample(trainX,trainY)
    testXUnder,testYUnder = undersample.fit_resample(testX, testY)






    # Explica lo que se hace en este paso
    clf = KNeighborsClassifier(n_neighbors=5,
                          weights='uniform',
                          algorithm='auto',
                          leaf_size=30,
                          p=2)

    # Explica lo que se hace en este paso



    clf.class_weight = "balanced"

    # Explica lo que se hace en este paso

    clf.fit(trainX, trainY)


# Build up our result dataset

# The model is now trained, we can apply it to our test set:
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    print(testX.head(5))
    print(testX.index)
    print(probas)

    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    cols = [
        u'probability_of_value_%s' % label
        for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
    ]
    probabilities = pd.DataFrame(data=probas, index=testX.index, columns=cols)

# Build scored dataset
    results_test = testX.join(predictions, how='left')
    results_test = results_test.join(probabilities, how='left')
    results_test = results_test.join(test['__target__'], how='left')
    results_test = results_test.rename(columns= {'__target__': 'TARGET'})

    i=0
    for real,pred in zip(testY,predictions):
        print(real,pred)
        i+=1
        if i>5:
            break

    print(f1_score(testY, predictions, average=None))
    print(classification_report(testY,predictions))
    print(confusion_matrix(testY, predictions, labels=[1,0]))
print("bukatu da")