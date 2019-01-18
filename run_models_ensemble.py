import json

import sys
import pandas as pd
import numpy as np
import prep
import os
import pdb
import argparse
import argcomplete

#Suppress messages from tensorflow
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import model_from_json, load_model
sys.stderr = stderr
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Stop TensorFlow Debugging Info


def open_model(model_file,x_test):

    model = load_model(model_file)
    y_pred = model.predict(x_test,batch_size = 500,verbose = 0)

    return y_pred


def print_error_for_model(y_pred,y_test,model_name_str):

    y_pred = np.concatenate((y_pred[0], y_pred[1],y_pred[2],y_pred[3]), axis=1)
    error_rounded =  (np.round(abs(y_pred-y_test),3))
    mean_error_rounded = np.mean(error_rounded,axis = 0)


    np.set_printoptions(precision=3)
    
    print('Results for: ' + model_name_str)
    for i,val in enumerate(mean_error_rounded):
        print("Mean Error Rounded for y{} is {:.2f}".format(i,val))
    print("Mean Error Rounded total is {:.2f}".format(np.mean(mean_error_rounded)))



def main(args):
    #model = load_model('my_model.h5')
    models_file = args.models_folder
    dataset = args.dataset



    df = pd.read_pickle(os.path.join("data",dataset + ".pkl"))

    x_test,y_test,y0_test,y1_test,y2_test,y3_test = prep.encode(df, False,0) 

    y_preds = []
    models_list = os.listdir(models_file)

    for i,model_name in enumerate(models_list):
        print("Loading Model "+ str(i)+'/'+str(len(models_list)))
        y_signle = open_model(os.path.join(models_file,model_name),x_test)
        print_error_for_model(y_signle,y_test,'Model:'+str(i)+'/'+str(len(models_list)))
        y_preds.append(y_signle)

    y_pred_total = np.zeros_like(y_preds[0])


    for i,y_pred in enumerate(y_preds):
        y_pred_total += y_pred


    y_pred_total = y_pred_total/len(y_preds) #mean
    print_error_for_model(y_pred_total,y_test,'Ensemple of '+str( len(y_preds)))



"""     #SAVE TO x_test + y_text in pandas and csv.

    wavelength_list = [0.46,0.54,0.7]
    total_pred = np.empty((y_pred.shape[0],7), float)

    for i,wavelength in enumerate(wavelength_list):

        lambda_pred = np.concatenate((np.expand_dims(x_test[:,i], axis=1),np.expand_dims(x_test[:,i+3], axis=1),y_pred),axis = 1)  
        w = np.empty((lambda_pred.shape[0],1)); w.fill(wavelength)
        lambda_pred = np.concatenate((lambda_pred,w),axis = 1)
        total_pred = np.append(total_pred,lambda_pred,axis = 0)

    #total_pred = np.round(total_pred,6)
    

    columns = ["Transmision","Phase","thickness","Radius","Pereodicity_X","Pereodicity_Y","lambda"]
    df_pred = pd.DataFrame(total_pred, columns=columns)
    df_pred.drop_duplicates(inplace=True)



    model_name_with_ext = os.path.basename(model_file)
    model_name, _ = os.path.splitext(model_name_with_ext)

    df_pred_name = dataset + "_"+ model_name

    df_pred.to_pickle(os.path.join("models","export_data",df_pred_name + ".pkl"))
    df_pred.to_csv(os.path.join("models","export_data",df_pred_name + ".csv"),index = False,sep = ",")

    #test = pd.read_pickle(os.path.join("models","export_data",df_pred_name + ".pkl"))
    #pdb.set_trace()
    print("Saved pkl and CSV file for model: "+ df_pred_name) """




    


    



if __name__ == "__main__":
	
    parser = argparse.ArgumentParser(description = 'LOads Model and DataFile')
    parser.add_argument('-mf','--models_folder',type = str, help='models folder location')
    parser.add_argument('-d','--dataset', type = str ,help = "dev/test/train",choices=['dev', 'test', 'train'])
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    main(args)