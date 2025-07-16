import kuramoto as kuramoto
from kuramoto import Kuramoto


import numpy as np
#from numba import njit, prange
import pandas as pd
from typing import Union, Optional, Tuple, Callable


# N:networksize,





def fc(ts):
    """Functional connectivity matrix of timeseries multidimensional `ts` (Nxt).
    Pearson correlation (from `np.corrcoef()` is used).

    :param ts: Nxt timeseries
    :type ts: numpy.ndarray
    :return: N x N functional connectivity matrix
    :rtype: numpy.ndarray
    """
    fc = np.corrcoef(ts)
    fc = np.nan_to_num(fc)  # remove NaNs
    return fc
##########################################################

def initial_angles(low,
                   high,
                   size,
                   num_trials):
    all_angles=[]
    for i in range(num_trials):
            all_angles.append(np.random.uniform(low, high, size))
    return all_angles        
#################################################################

def from_SC_to_FC(graph,
                  freq,
                  N,
                  runtime,
                  num_trials,
                  coupling_factor,
                  dt_int,
                  finalTime,
                  initial_angl,
                  Kuramoto:Callable = Kuramoto):
    # act_h=[]
    sin_act = np.zeros(shape=(N,runtime,num_trials))
    FCmat_c=[] # save FC that is in 3D for individual coupling factor
    for j in range(len(coupling_factor)):
        for i in range(num_trials):
    #         print(initial_angles)
            model = Kuramoto(coupling=coupling_factor[j], dt=dt_int, T=finalTime, n_nodes=len(graph), natfreqs=freq)
            activity = model.run(adj_mat=graph, angles_vec=initial_angl[i])
        #     act_h.append(act_mat)  
            sin_act[:,:,i]= np.sin(activity) # collect timeseries data
        # produce Functional connectivity
        fc_matrices= np.zeros(shape=(N,N,num_trials))
        for k in range(num_trials):
            fc_matrices[:,:,k] = fc(sin_act[:,:,k])
        FCmat_c.append(fc_matrices)

        # Creating histogram
        #distribution of synchrony
    #     a=fc_matrices[4,10,:]
    #     fig, ax = plt.subplots(figsize =(5, 3))
    #     ax.hist(a)
    #     plt.title('Healthy, c={}'.format(c[j]))
    #     plt.show()
    return FCmat_c
#########################################################################


def meanFC(coupling_factor,
           FCmat_c):
    mFC=[]
    for k in range(len(coupling_factor)):
        # mean of FC for individual coupling factor
        mFC.append(np.mean(FCmat_c[k],axis=2))
    return mFC

#         #plot FC matrix
#         plt.subplot(m,n,int(j)+1)   
#         sns.heatmap(mFC_pd[j])
#         plt.title('PD, average of FC over {} initial conditions, c={}'.format(num_trials,
#                                                                               coupling_factor[j]),fontsize = 6)
#         plt.tight_layout()
####################################################################################

def varFC(coupling_factor,
          FCmat_c):
    vFC=[]
    for k in range(len(coupling_factor)):
    # variance of FC for individual coupling factor
        vFC.append(np.var(FCmat_c[k],axis=2))
    return vFC




#########################################################################
# do flatten on each individual FC matrix

def flat_ind_FCmat(FCmat_c,
                   coupling_factor,
                   num_trials):
    all_flat=[]
#     all_flat_pd=[]
    for i in range(len(coupling_factor)):
        # for PCA for PD
        flat_FC=[]
#         flat_FC_pd=[]
        for j in range(num_trials):
            flat_FC.append(FCmat_c[i][:,:,j].flatten())
#             flat_FC_pd.append(FCmat_c_pd[i][:,:,j].flatten())
        all_flat.append(flat_FC)
#         all_flat_pd.append(flat_FC_pd)
    return all_flat


#########################################################################
# do flatten on each individual row in FC matrix

def flat_row_FCmat(num_row,
                   FCmat_c,
                   coupling_factor,
                   num_trials):
    all_flat=[]
#     all_flat_pd=[]
    for i in range(len(coupling_factor)):
        # for PCA for PD
        flat_FC=[]
#         flat_FC_pd=[]
        for j in range(num_trials):
            flat_FC.append(FCmat_c[i][num_row,:,j].flatten())
#             flat_FC_pd.append(FCmat_c_pd[i][:,:,j].flatten())
        all_flat.append(flat_FC)
#         all_flat_pd.append(flat_FC_pd)
    return all_flat
#########################################################################
# PCA on FC for both H and PD
# for individual coupling factor k



def PCA_FC(flat_FC_h,
          flat_FC_pd,
          coupling_factor,
          num_trials,
          num_features):

    import numpy as np
    from sklearn.decomposition import PCA
    
    principalDf2=[]
    for i in range(len(coupling_factor)):
#         # for PCA for PD
#         flat_FC_h=[]
#         flat_FC_pd=[]
#         for j in range(num_trials):
#             flat_FC_h.append(FCmat_c_h[i][:,:,j].flatten())
#             flat_FC_pd.append(FCmat_c_pd[i][:,:,j].flatten())


        df1=pd.DataFrame(flat_FC_h[i])
        df2=pd.DataFrame(flat_FC_pd[i])
        df1['class']='Healthy'
        df2['class']='PD'
        df= pd.concat([df1, df2])
        # df
        # X = pd.DataFrame(df)
        X= df.iloc[:,0:num_features]
        pca = PCA(n_components=2)
        # pca.fit(X)
        # print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)


        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['principal1', 'principal2'])
        principalDf1= pd.concat([principalDf, pd.DataFrame(df['class']).set_index(principalDf.index)],axis=1)
        principalDf2.append(principalDf1)
    # fig = plt.figure(figsize = (8,8))
#     sns.scatterplot(data=principalDf2 ,x='principal1',
#                 y='principal2',hue='class',alpha=0.4).set(title='PCA on Healthy and PD, c=0.1')

#     sns.jointplot(data=principalDf2 ,x='principal1',
#                 y='principal2',hue='class', kind="kde")

    return principalDf2

#######################################################
# PCA: dif rows for individual c_factor

def rows_in_each_C(c_order,rows,
                   num_trials,
                   num_features,
                   FCmat_c_h,
                   FCmat_c_pd,
                   coupling_factor,
                   function:Callable=flat_row_FCmat,):
    principalDf2=[]
    for i in range(len(rows)):
        num_row=rows[i]
        flat_h= function(num_row,FCmat_c_h,
                           coupling_factor,
                           num_trials)

        flat_pd= function(num_row,FCmat_c_pd,
                           coupling_factor,
                           num_trials)
        # PCA
        from sklearn.decomposition import PCA
        df1=pd.DataFrame(flat_h[c_order])
        df2=pd.DataFrame(flat_pd[c_order])
        df1['class']='Healthy'
        df2['class']='PD'
        df= pd.concat([df1, df2])
        # df
        # X = pd.DataFrame(df)
        num_features=12
        X= df.iloc[:,0:num_features]
        pca = PCA(n_components=2)
        # pca.fit(X)
        # print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)


        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns = ['principal1', 'principal2'])
        principalDf1= pd.concat([principalDf, pd.DataFrame(df['class']).set_index(principalDf.index)],axis=1)
        principalDf2.append(principalDf1)
    return principalDf2

###########################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()


import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
        
