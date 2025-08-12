import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm
import numpy as np

filepath = 'C:/Users/benvi/OneDrive - University of Illinois - Urbana/GMF/ATA 36 Analysis/PCA app/input.xlsx'
outputpath = 'C:/Users/benvi/OneDrive - University of Illinois - Urbana/GMF/ATA 36 Analysis/PCA app/output2.xlsx'
sheet = 'corrected feature matrix'
class PCAModel:
    def __init__(self):
        pass
    def train_pca(self,df, components, colnames, fails, healthy):
        try:
            print("read success")

            scalers = StandardScaler()
            scaler_tosave = scalers.fit(df)
            print(f"scaler means: {scaler_tosave.mean_}")
            print(f"scaler standevs: {scaler_tosave.scale_}")

            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(df)

            #making the model
            self.pca_model = PCA(n_components=components)
            pca_components = self.pca_model.fit_transform(scaled_features)
            
            #pca_params = pd.DataFrame(data=self.pca_model.components_, columns=[f"{i} loading" for i in colnames])
            """pca_params = pd.DataFrame([[f"{i} loading" for i in colnames],
                                       self.pca_model.components_.tolist(),
                                       [np.nan],
                                       [f"mean of {i}" for i in colnames],
                                       scaler_tosave.mean_.tolist(),
                                       [np.nan],
                                       [f"standev of {i}" for i in colnames],
                                       scaler_tosave.scale_.tolist(),
                                       [np.nan]])"""
            pca_params= []
            comp_num =1
            for i in self.pca_model.components_:
                pca_params.append(np.append(i, f"PC{comp_num} loading"))
                comp_num +=1

            failset = pca_components[fails[0]:fails[1]]
            healthset = pca_components[healthy[0]:healthy[1]]
            print(failset)
            print(healthset)
           
            
            fail_param= pd.DataFrame([np.append([*skewnorm.fit(failset[:,i])], f"PC{i+1} fail skewnorm") for i in range(components)])
            health_param= pd.DataFrame([np.append([*skewnorm.fit(healthset[:,i])], f"PC{i+1} health skewnorm") for i in range(components)])
            
            
            
            

            pca_params.append([np.nan]*len(colnames))
            pca_params.append(np.append(scaler_tosave.mean_, "scaler mean"))
            pca_params.append([np.nan]*len(colnames))
            pca_params.append(np.append(scaler_tosave.scale_, "scaler standard deviation"))
            pca_params.append([np.nan]*len(colnames))
            
            #print(pca_params)
            comp_rows=pd.DataFrame(pca_params)
            
            buff= pd.DataFrame([[np.nan]*len(colnames)])
            df_comp_rows = pd.concat([comp_rows, buff, fail_param, buff, health_param], axis=0)
            #print(comp_rows)
            
            #prob dist
            
           
            

          
            
            #params
            #print(self.pca_model.components_.shape)
            buffer = pd.DataFrame({" ":[""]*4})
            
            pca_df = pd.DataFrame(data=pca_components, columns=[f"PC{i}" for i in range(1,components+1)])
            
            pca_df.reset_index(drop=True, inplace=True)
            buffer.reset_index(drop=True, inplace=True)
            df_comp_rows.reset_index(drop=True, inplace=True)
            tooutput = pd.concat([pca_df,buffer, df_comp_rows], axis=1)
            return tooutput


        except FileNotFoundError as e:
            print("filenotfound")
            return
    
    def get_pcascore(self, data):
        if self.pca_model is None or self.scaler is None:
            raise ValueError("Model has not been trained. Call train_pca() first.")
        scaled_features = self.scaler.transform(data)
        return self.pca_model.transform(scaled_features)

    def inverse_transform(self, scores):
        # Convert to numpy array if it's not already
        scores = np.array(scores)
        
        # Handle 1D input (single sample)
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        
        # Ensure we have the right number of components
        if scores.shape[1] != self.pca_model.n_components_:
            raise ValueError(
                f"Expected {self.pca_model.n_components_} components, "
                f"got {scores.shape[1]}"
            )
        
        # Transform back to original feature space
        scaled_features = self.pca_model.inverse_transform(scores)
        
        # Inverse the scaling
        original_features = self.scaler.inverse_transform(scaled_features)
        features_df = pd.DataFrame(original_features)
        return features_df
pca = PCAModel()
try:
    pass

except Exception as e:
    print(str(e))
    

#df = pd.read_excel(filepath, usecols=range(0,4), header=None)
    
"""outputexcel = pca.train_pca(df,2,["mean", 'standard deviation', 'slope', 'mean engine diff'], (0,112), (112,224))
outputexcel.to_excel(outputpath, index=False)"""
        
