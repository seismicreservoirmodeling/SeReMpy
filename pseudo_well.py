import numpy as np
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt 
from Geostats import *

class PseudoWell(object):

    def simulate_properties(self, d_prop, depth_simulation):
        
        # get configuration from dictionary
        n_ptos = depth_simulation.shape[0]
        n_prop = len(d_prop['name'])
        types = d_prop['vartype']
        ranges = d_prop['range']
        sills = d_prop['sill']
        nugget = d_prop['nugget']
        means = d_prop['means']        

        # check if there is conditional points
        if 'cond_depth' in d_prop.keys():
            cond_depth = d_prop['cond_depth']
            cond_values = d_prop['cond_values']
        else:
            cond_depth = np.zeros((0,1))
            cond_values = np.zeros((0,1))
    
        #  Use only the data conditioning that are within on the simulation depth range
        indices = np.nonzero( (cond_depth >= depth_simulation[0]) & (cond_depth <= depth_simulation[-1]) )[0]
        cond_depth = cond_depth[indices]        

        # run SGSIM    
        simulation = np.zeros((depth_simulation.shape[0], n_prop))
        for prop in range(n_prop):            
            if 'cond_depth' in d_prop.keys():
                cond_values_ = cond_values[prop]
                simulation[:,prop] = SeqGaussianSimulation(depth_simulation, cond_depth, cond_values_[indices], means[prop], sills[prop], ranges[prop], types[prop], 0).reshape((n_ptos))
            else:
                simulation[:,prop] = SeqGaussianSimulation(depth_simulation, cond_depth, cond_values, means[prop], sills[prop], ranges[prop], types[prop], 0).reshape((n_ptos))
            
        return simulation

    def simulate_facies(self, df_facies, depth_simulation):    
            
        # get configuration from dictionary
        n_facies = len(df_facies['name'])
        facies_labels = np.arange(n_facies)
        ranges = df_facies['range']
        types = df_facies['vartype']
        proportions = df_facies['prop']   

        # check if there is conditional points
        if 'cond_depth' in df_facies.keys():
            cond_depth = df_facies['cond_depth']
            cond_values = df_facies['cond_values']
        else:
            cond_depth = np.zeros((0,1))
            cond_values = np.zeros((0,1))

        #  Use only the data conditioning that are within on the simulation depth range
        indices = np.nonzero( (cond_depth >= depth_simulation[0]) & (cond_depth <= depth_simulation[-1]) )[0]
        cond_depth = cond_depth[indices]
        cond_values = cond_values[indices]           
        
        facies = SeqIndicatorSimulation(depth_simulation, cond_depth, cond_values, n_facies, proportions, ranges, types)
        
        return facies

    def simulate_pseudo_well(self, list_simulation_configuration: list):
    # TO DO simulation without conditioning data, miguezinho:    
            
        n_macrolayers = len(list_simulation_configuration['name'])
        df_facies_all = list_simulation_configuration['facies_config']
        depths_all = list_simulation_configuration['DEPTH']
        n_prop = len(df_facies_all[0]['properties_configs'][0]['name'])
        
        facies_pseudowell = []
        properties_pseudowell = np.zeros((0, n_prop))
        
        depth = []
        for macrolayer in range(n_macrolayers):        

            facies_macrolayer = self.simulate_facies(df_facies_all[macrolayer],depths_all[macrolayer])
            
            properties_macrolayer = np.zeros((depths_all[macrolayer].shape[0], n_prop))
            for facies in range(len(df_facies_all[macrolayer]['name'])):
                properties_simulation = self.simulate_properties(df_facies_all[macrolayer]['properties_configs'][facies],depths_all[macrolayer])
                properties_macrolayer[np.nonzero(facies_macrolayer==facies)[0],:] = properties_simulation[np.nonzero(facies_macrolayer==facies)[0],:]

            facies_pseudowell = np.append(facies_pseudowell, facies_macrolayer )
            depth = np.append(depth, depths_all[macrolayer])
            properties_pseudowell = np.concatenate((properties_pseudowell, properties_macrolayer), axis=0)
        
        # write pseudo well in a regular dictionary structure
            
        pseudo_well = {'DEPTH': depth, 'FACIES': facies_pseudowell}

        for prop in range(n_prop):            
            pseudo_well[df_facies_all[0]['properties_configs'][0]['name'][prop]] = properties_pseudowell[:,prop]
            
        return pseudo_well

if __name__ == '__main__':
        
    ################## TEST PROPERTY STRUCT OF MACROLAYER ####################            
    
    # Depth for simulation
    n_ptos = 500
    depth_simulation = np.arange(0.0,n_ptos).reshape((n_ptos,1))*0.15 + 3000

    mean_phi = np.linspace(0.05,0.25,n_ptos).reshape((n_ptos,1))
    mean_sw = 1 - np.linspace(0.05,0.25,n_ptos).reshape((n_ptos,1))

    n_cond_ptos = 10
    cond_depth = depth_simulation

    # Conditional points    
    random_indices = np.random.choice(n_ptos, size=n_cond_ptos, replace=False)
    cond_depth = depth_simulation[random_indices, :]
    cond_depth[-1] = depth_simulation[-1] + 2
    cond_phi = mean_phi[random_indices,:] + 0.02*np.random.randn(n_cond_ptos,1)
    cond_sw = mean_sw[random_indices,:] + 0.02*np.random.randn(n_cond_ptos,1)

    df_prop_fc1 = {'name': ['porosity', 'saturation'], 'vartype': ['gau','gau'], 'range': [3.0,1.0], 'sill': [0.01,0.005], 'nugget': [0.0,0.0], 'means': [mean_phi, mean_sw], 'cond_depth': cond_depth, 'cond_values': [cond_phi, cond_sw] }
    

    pw = PseudoWell()
    properties = pw.simulate_properties(df_prop_fc1,depth_simulation)
    
    plt.plot(depth_simulation,properties)
    plt.plot(cond_depth, cond_phi,'o')
    plt.plot(cond_depth, cond_sw,'o')
    plt.show()
    
    ######################## TEST PSEUDO WELL STRUCT ##########################        
    step = 0.15
    n_ptos_z1 = 700
    depth_z1 = np.arange(0.0,n_ptos_z1).reshape((n_ptos_z1,1))*step + 1000
    n_ptos_z2 = 1100
    depth_z2 = np.arange(0.0,n_ptos_z2).reshape((n_ptos_z2,1))*step + depth_z1[-1] + step
    n_ptos_z3 = 800
    depth_z3 = np.arange(0.0,n_ptos_z3).reshape((n_ptos_z3,1))*step + depth_z2[-1] + step
    depth_pseudowell = np.concatenate((depth_z1, depth_z2,depth_z3), axis=0)

    # Local variable mean IN ZONE/MACROLAYER 1 
    mean_phi_sand_z1 = np.linspace(0.05,0.25,n_ptos_z1).reshape((n_ptos_z1,1))
    mean_sw_sand_z1 = 1 - np.linspace(0.05,0.25,n_ptos_z1).reshape((n_ptos_z1,1))

    mean_phi_shale_z1 = np.linspace(0.05,0.15,n_ptos_z1).reshape((n_ptos_z1,1))
    mean_sw_shale_z1 = 1 - np.linspace(0.05,0.85,n_ptos_z1).reshape((n_ptos_z1,1))

    mean_phi_carb_z1 = np.linspace(0.05,0.25,n_ptos_z1).reshape((n_ptos_z1,1))
    mean_sw_carb_z1 = 1 - np.linspace(0.05,0.55,n_ptos_z1).reshape((n_ptos_z1,1))

    # HARD DATA POINTS
    n_cond_ptos = 30
    random_indices = np.random.choice(n_ptos_z1+n_ptos_z2+n_ptos_z3, size=n_cond_ptos, replace=False)
    cond_depth = depth_pseudowell[random_indices, :]
    cond_phi = 0.1 + 0.02*np.random.randn(n_cond_ptos,1)
    cond_sw = 0.8 + 0.02*np.random.randn(n_cond_ptos,1)
    cond_facies = np.random.randint(3, size=n_cond_ptos).reshape((n_cond_ptos,1))

    ## define struct of pseudo well

    # with data conditioning
    # zone 1 (with trend)
    d_prop_sand_z1 = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [10.0,2.0], 'sill': [0.01,0.005], 'nugget': [0.0,0.0], 'means': [mean_phi_sand_z1, mean_sw_sand_z1], 'cond_depth': cond_depth , 'cond_values': [cond_phi, cond_sw] }
    d_prop_shale_z1 = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [30.0,3.0], 'sill': [0.001,0.001], 'nugget': [0.0,0.0], 'means': [mean_phi_shale_z1, mean_sw_shale_z1], 'cond_depth': cond_depth , 'cond_values': [cond_phi, cond_sw] }
    d_prop_carb_z1 = {'name': ['porosity', 'saturation'], 'vartype': ['exp','exp'], 'range': [10.0,1.0], 'sill': [0.001,0.005], 'nugget': [0.0,0.0], 'means': [mean_phi_carb_z1, mean_sw_carb_z1], 'cond_depth': cond_depth , 'cond_values': [cond_phi, cond_sw] }        
    # other zones (without trend)
    d_prop_sand = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [10.0,2.0], 'sill': [0.01,0.005], 'nugget': [0.0,0.0], 'means': [0.25, 0.2], 'cond_depth': cond_depth , 'cond_values': [cond_phi, cond_sw] }
    d_prop_shale = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [30.0,3.0], 'sill': [0.001,0.001], 'nugget': [0.0,0.0], 'means': [0.05, 0.9], 'cond_depth': cond_depth , 'cond_values': [cond_phi, cond_sw] }
    d_prop_carb = {'name': ['porosity', 'saturation'], 'vartype': ['exp','exp'], 'range': [10.0,1.0], 'sill': [0.001,0.005], 'nugget': [0.0,0.0], 'means': [0.15, 0.2], 'cond_depth': cond_depth , 'cond_values': [cond_phi, cond_sw] }

    d_facies_M1 = {'name': ['sand', 'shale','carb'], 'prop': [0.8, 0.2, 0.000001], 'vartype': ['exp','exp','exp'], 'range': [30.0, 10.0, 1.0], 'properties_configs': [d_prop_sand_z1, d_prop_shale_z1, d_prop_carb_z1],'cond_depth': cond_depth ,'cond_values':cond_facies}
    d_facies_M2 = {'name': ['sand', 'shale','carb'], 'prop': [0.45, 0.1, 0.45], 'vartype': ['exp','exp','exp'], 'range': [30.0, 1.0, 30.0], 'properties_configs': [d_prop_sand, d_prop_shale, d_prop_carb],'cond_depth': cond_depth ,'cond_values':cond_facies}
    d_facies_M3 = {'name': ['sand', 'shale','carb'], 'prop': [0.95, 0.05, 0.000001], 'vartype': ['exp','exp','exp'], 'range': [100.0, 1.0, 1.0], 'properties_configs': [d_prop_sand, d_prop_shale, d_prop_carb],'cond_depth': cond_depth ,'cond_values':cond_facies}

    # without data conditioning
    # zone 1 (with trend)
    #d_prop_sand_z1 = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [10.0,2.0], 'sill': [0.01,0.005], 'nugget': [0.0,0.0], 'means': [mean_phi_sand_z1, mean_sw_sand_z1]}
    #d_prop_shale_z1 = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [30.0,3.0], 'sill': [0.001,0.001], 'nugget': [0.0,0.0], 'means': [mean_phi_shale_z1, mean_sw_shale_z1]}
    #d_prop_carb_z1 = {'name': ['porosity', 'saturation'], 'vartype': ['exp','exp'], 'range': [10.0,1.0], 'sill': [0.001,0.005], 'nugget': [0.0,0.0], 'means': [mean_phi_carb_z1, mean_sw_carb_z1]}
    # other zones (without trend)
    #d_prop_sand = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [10.0,2.0], 'sill': [0.01,0.005], 'nugget': [0.0,0.0], 'means': [0.25, 0.2]}
    #d_prop_shale = {'name': ['porosity', 'saturation'], 'vartype': ['sph','sph'], 'range': [30.0,3.0], 'sill': [0.001,0.001], 'nugget': [0.0,0.0], 'means': [0.05, 0.9]}
    #d_prop_carb = {'name': ['porosity', 'saturation'], 'vartype': ['exp','exp'], 'range': [10.0,1.0], 'sill': [0.001,0.005], 'nugget': [0.0,0.0], 'means': [0.15, 0.2] }

    #d_facies_M1 = {'name': ['sand', 'shale','carb'], 'prop': [0.8, 0.2, 0.000001], 'vartype': ['exp','exp','exp'], 'range': [30.0, 10.0, 1.0], 'properties_configs': [d_prop_sand_z1, d_prop_shale_z1, d_prop_carb_z1]}
    #d_facies_M2 = {'name': ['sand', 'shale','carb'], 'prop': [0.45, 0.1, 0.45], 'vartype': ['exp','exp','exp'], 'range': [30.0, 1.0, 30.0], 'properties_configs': [d_prop_sand, d_prop_shale, d_prop_carb]}
    #d_facies_M3 = {'name': ['sand', 'shale','carb'], 'prop': [0.95, 0.05, 0.000001], 'vartype': ['exp','exp','exp'], 'range': [100.0, 1.0, 1.0], 'properties_configs': [d_prop_sand, d_prop_shale, d_prop_carb]}

    # PSEUDO WELL DICT
    d = {'name': ['zone1', 'zone2','zone3'], 'DEPTH': [depth_z1, depth_z2, depth_z3], 'facies_config': [d_facies_M1, d_facies_M2, d_facies_M3]}
     
    pseudo_well = pw.simulate_pseudo_well(d)
    
    plt.subplot(2, 1, 1)
    plt.plot(pseudo_well['DEPTH'],pseudo_well['porosity'])
    plt.plot(pseudo_well['DEPTH'],pseudo_well['saturation'])    
    plt.plot(cond_depth,cond_phi,'o')
    plt.plot(cond_depth,cond_sw,'o')
    plt.axvline(x=depth_z2[0],color='r')
    plt.axvline(x=depth_z2[-1],color='r')    
    plt.subplot(2, 1, 2)
    plt.plot(pseudo_well['DEPTH'],pseudo_well['FACIES'])
    plt.plot(cond_depth,cond_facies,'o')
    plt.axvline(x=depth_z2[0],color='r')
    plt.axvline(x=depth_z2[-1],color='r')
    plt.show()
        
    
    