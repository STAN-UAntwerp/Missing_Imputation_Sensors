import warnings
from methods.BRITS.model import BRITS
from methods.splineImputation import apply_spline
warnings.filterwarnings("ignore")
import pandas as pd 

def BRITS_imputation(df_missings,df_ground_truth,**kwargs):
    """
        BRITS main function.
    
    Args:
        - seq_len: sequence length of time-series data
        - h_dim: hidden state dimensions
        - batch_size: the number of samples in mini batch
        - epochs: the number of epochs
        - learning_rate: learning rate of model training
        
    Returns:
        - output:
        - imputed_x: imputed data
    """  
    epochs = kwargs.get('epochs', 100)
    batch_size = kwargs.get('batch_size', 1)
    h_dim = kwargs.get('h_dim', 10)
    timesteps = kwargs.get('timesteps', 20)
    learning_rate = kwargs.get('learning_rate', 0.01)

    # Transform the data
    pivot_df_ms = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    pivot_df_gt = df_ground_truth.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    pivot_df_ms = pivot_df_ms.droplevel(level='latitude', axis=1).reset_index(drop=True)
    pivot_df_gt = pivot_df_gt.droplevel(level='latitude', axis=1).reset_index(drop=True)
    pivot_df_ms.columns = [str(col) for col in pivot_df_ms.columns]
    pivot_df_gt.columns = [str(col) for col in pivot_df_gt.columns]
    x = pivot_df_gt.to_numpy()
    xm = pivot_df_ms.to_numpy()

    # Fit the model
    model = BRITS(x=x,units=h_dim,timesteps=timesteps)
    model.fit(learning_rate=learning_rate,batch_size=batch_size,epochs=epochs,verbose=False)

    # Run it on the missings 
    x_hat = model.impute(x=xm)

    # Output to correct format
    imputed_df = pd.DataFrame(x_hat)
    pivot_df_ms = df_missings.pivot(index='time', columns=['longitude', 'latitude'], values='target')
    imputed_df.columns = pivot_df_ms.columns
    if imputed_df.shape[0] < pivot_df_ms.shape[0]:
        imputed_df.index = pivot_df_ms.index[pivot_df_ms.shape[0]-imputed_df.shape[0]:]
        imputed_df =  pd.concat([imputed_df,pivot_df_ms[:imputed_df.index[0]-1]])
    else:
        imputed_df.index = pivot_df_ms.index
    imputed_df.sort_index(inplace=True)
    unpivoted_df = imputed_df.reset_index().melt(id_vars='time', var_name=['longitude', 'latitude'], value_name='target')
    unpivoted_df = apply_spline(unpivoted_df,unpivoted_df)

    return unpivoted_df