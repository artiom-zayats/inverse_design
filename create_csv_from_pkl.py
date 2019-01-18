import numpy as np
import pandas as pd
import os




def main(dataset):
    df = pd.read_pickle(os.path.join("data",dataset+".pkl"))
    import pdb; pdb.set_trace()

    df[""]

    #df.rename(index=str,columns = {'A_x':'A_1','A_y':'A_2','phi_x':'phi_1','phi_y':'phi_2','lambda_x':'lambda_1','lambda_y':'lambda_2'},inplace = True)
    


if __name__ == "__main__":
	main("dev")