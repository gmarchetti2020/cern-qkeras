import numpy as np
import math
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
#import tensorflow_probability as tfp
from qkeras import QDense, QActivation

#tf.compat.v1.enable_eager_execution()

def preprocess_anomaly_data(pT_scaler, anomaly_data):
    anomaly_data[:,9:19,0] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,0])
    anomaly_data[:,9:19,0] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,0])
    anomaly_data[:,9:19,1] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,1])
    anomaly_data[:,9:19,1] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,1])
    anomaly_data[:,9:19,2] = np.where(anomaly_data[:,9:19,1]>4,0,anomaly_data[:,9:19,2])
    anomaly_data[:,9:19,2] = np.where(anomaly_data[:,9:19,1]<-4,0,anomaly_data[:,9:19,2])
    
    data_noMET = anomaly_data[:,1:,:]
    MET = anomaly_data[:,0,[0,2]]

    pT = data_noMET[:,:,0]
    eta = data_noMET[:,:,1]
    phi = data_noMET[:,:,2]

    pT = np.concatenate((MET[:,0:1],pT), axis=1) # add MET pt for scaling
    mask_pT = pT!=0

    pT_scaled = np.copy(pT)
    pT_scaled = pT_scaler.transform(pT_scaled)
    pT_scaled = pT_scaled*mask_pT

    phi = np.concatenate((MET[:,1:2], phi), axis=1)

    test_scaled = np.concatenate((pT_scaled[:,0:1], pT_scaled[:,1:], eta, phi), axis=1)
    test_notscaled = np.concatenate((MET[:,0:1], data_noMET[:,:,0], eta, phi), axis=1)
    
    return test_scaled, test_notscaled


def custom_loss_negative(true, prediction):
    
    #mse_loss = tf.keras.losses.MeanSquaredError()
    # 0-1 = met(pt,phi) , 2-14 = egamma, 14-26 = muon, 26-56 = jet; (pt,eta,phi) order
    #MASK PT
    mask_met = tf.math.not_equal(true[:,0:1],0)
    mask_met = tf.cast(mask_met, tf.float32)
    mask_eg = tf.math.not_equal(true[:,1:5],0)
    mask_eg = tf.cast(mask_eg, tf.float32)
    mask_muon = tf.math.not_equal(true[:,5:9],0)
    mask_muon = tf.cast(mask_muon, tf.float32)
    mask_jet = tf.math.not_equal(true[:,9:19],0)
    mask_jet = tf.cast(mask_jet, tf.float32)

    # PT
    met_pt_pred = tf.math.multiply(prediction[:,0:1],mask_met) #MET
    jets_pt_pred = tf.math.multiply(prediction[:,9:19],mask_jet) #Jets
    muons_pt_pred = tf.math.multiply(prediction[:,5:9],mask_muon) #Muons
    eg_pt_pred = tf.math.multiply(prediction[:,1:5],mask_eg) #EGammas
    
    # ETA
    jets_eta_pred = tf.math.multiply(4.0*(tf.math.tanh(prediction[:,27:37])),mask_jet) #Jets
    muons_eta_pred = tf.math.multiply(2.1*(tf.math.tanh(prediction[:,23:27])),mask_muon) #Muons
    eg_eta_pred = tf.math.multiply(3.0*(tf.math.tanh(prediction[:,19:23])),mask_eg) #EGammas
    
    # PHI
    met_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,37:38]),mask_met) #MET
    jets_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,46:56]),mask_jet) #Jets
    muon_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,42:46]),mask_muon) #Muons
    eg_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,38:42]),mask_eg) #EGammas
    
    y_pred = tf.concat([met_pt_pred, eg_pt_pred, muons_pt_pred, jets_pt_pred, eg_eta_pred, muons_eta_pred, jets_eta_pred,\
                       met_phi_pred, eg_phi_pred, muon_phi_pred, jets_phi_pred], axis=-1)
    loss = tf.reduce_mean(tf.math.square(true - y_pred),axis=-1)
    return -loss

def custom_loss_training(true, prediction):
    
    #mse_loss = tf.keras.losses.MeanSquaredError()
    # 0-1 = met(pt,phi) , 2-14 = egamma, 14-26 = muon, 26-56 = jet; (pt,eta,phi) order
    #MASK PT
    mask_met = tf.math.not_equal(true[:,0:1],0)
    mask_met = tf.cast(mask_met, tf.float32)
    mask_eg = tf.math.not_equal(true[:,1:5],0)
    mask_eg = tf.cast(mask_eg, tf.float32)
    mask_muon = tf.math.not_equal(true[:,5:9],0)
    mask_muon = tf.cast(mask_muon, tf.float32)
    mask_jet = tf.math.not_equal(true[:,9:19],0)
    mask_jet = tf.cast(mask_jet, tf.float32)

    # PT
    met_pt_pred = tf.math.multiply(prediction[:,0:1],mask_met) #MET
    jets_pt_pred = tf.math.multiply(prediction[:,9:19],mask_jet) #Jets
    muons_pt_pred = tf.math.multiply(prediction[:,5:9],mask_muon) #Muons
    eg_pt_pred = tf.math.multiply(prediction[:,1:5],mask_eg) #EGammas
    
    # ETA
    jets_eta_pred = tf.math.multiply(4.0*(tf.math.tanh(prediction[:,27:37])),mask_jet) #Jets
    muons_eta_pred = tf.math.multiply(2.1*(tf.math.tanh(prediction[:,23:27])),mask_muon) #Muons
    eg_eta_pred = tf.math.multiply(3.0*(tf.math.tanh(prediction[:,19:23])),mask_eg) #EGammas
    
    # PHI
    met_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,37:38]),mask_met) #MET
    jets_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,46:56]),mask_jet) #Jets
    muon_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,42:46]),mask_muon) #Muons
    eg_phi_pred = tf.math.multiply(math.pi*tf.math.tanh(prediction[:,38:42]),mask_eg) #EGammas
    
    y_pred = tf.concat([met_pt_pred, eg_pt_pred, muons_pt_pred, jets_pt_pred, eg_eta_pred, muons_eta_pred, jets_eta_pred,\
                       met_phi_pred, eg_phi_pred, muon_phi_pred, jets_phi_pred], axis=-1)
    loss = tf.reduce_mean(tf.math.square(true - y_pred),axis=-1)
    return loss

def mse_loss(inputs, outputs):
    return np.mean((inputs-outputs)*(inputs-outputs), axis=-1)

def custom_loss_numpy(true, prediction):
    #mse_loss = tf.keras.losses.MeanSquaredError()
    # 0-1 = met(pt,phi) , 2-14 = egamma, 14-26 = muon, 26-56 = jet; (pt,eta,phi) order
    #MASK PT
    mask_met = np.not_equal(true[:,0:1],0)
    mask_eg = np.not_equal(true[:,1:5],0)
    mask_muon = np.not_equal(true[:,5:9],0)
    mask_jet = np.not_equal(true[:,9:19],0)

    # PT
    met_pt_pred = np.multiply(prediction[:,0:1],mask_met) #MET
    jets_pt_pred = np.multiply(prediction[:,9:19],mask_jet) #Jets
    muons_pt_pred = np.multiply(prediction[:,5:9],mask_muon) #Muons
    eg_pt_pred = np.multiply(prediction[:,1:5],mask_eg) #EGammas
    
    # ETA
    jets_eta_pred = np.multiply(4.0*(np.tanh(prediction[:,27:37])),mask_jet) #Jets
    muons_eta_pred = np.multiply(2.1*(np.tanh(prediction[:,23:27])),mask_muon) #Muons
    eg_eta_pred = np.multiply(3.0*(np.tanh(prediction[:,19:23])),mask_eg) #EGammas
    
    # PHI
    met_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,37:38]),mask_met) #MET
    jets_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,46:56]),mask_jet) #Jets
    muon_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,42:46]),mask_muon) #Muons
    eg_phi_pred = np.multiply(math.pi*np.tanh(prediction[:,38:42]),mask_eg) #EGammas
    
    y_pred = np.concatenate([met_pt_pred, eg_pt_pred, muons_pt_pred, jets_pt_pred, eg_eta_pred, muons_eta_pred, jets_eta_pred,\
                       met_phi_pred, eg_phi_pred, muon_phi_pred, jets_phi_pred], axis=-1)
    loss = mse_loss(true,y_pred)
    return loss


def roc_objective(ae, X_test, bsm_data):
    def roc_objective_val(y_true, y_pred):
        # evaluate mse term
        predicted_qcd = ae(X_test, training=False)
        #mse_qcd = custom_loss_numpy(X_test, predicted_qcd.numpy()) ## THIS IS WHERE WE REQUIRE EAGER EXECUTION ##
        mse_qcd = custom_loss_training(X_test, predicted_qcd) ## THIS IS WHERE WE REQUIRE EAGER EXECUTION ##

        predicted_bsm = ae(bsm_data, training=False)
        #mse_bsm = custom_loss_numpy(bsm_data, predicted_bsm.numpy())
        mse_bsm = custom_loss_training(bsm_data, predicted_bsm)

        #mse_true_val = np.concatenate((np.ones(bsm_data.shape[0]), np.zeros(X_test.shape[0])), axis=-1)
        mse_true_val = tf.concat([tf.ones(bsm_data.shape[0]), tf.zeros(X_test.shape[0])], axis=-1)
        #mse_pred_val = np.concatenate((mse_bsm, mse_qcd), axis=-1)
        mse_pred_val=tf.concat([mse_bsm, mse_qcd], axis=-1)
        #mse_fpr_loss, mse_tpr_loss, mse_threshold_loss = roc_curve(mse_true_val, mse_pred_val)
        mse_fpr_loss, mse_tpr_loss, mse_threshold_loss = roc_curve(mse_true_val.numpy(), mse_pred_val.numpy())
        
        mse_objective = np.interp(10**(-5), mse_fpr_loss, mse_tpr_loss)
        
    
        # WITH TF OPERATIONS (NO EAGER MODE)
        #m = tf.keras.metrics.SensitivityAtSpecificity(specificity=1-(10**(-5)))
        #mse_pred_val_np_norm = mse_pred_val_np / mse_pred_val_np.max()
        #m.update_state(mse_true_val_np, mse_pred_val_np_norm)
        #mse_objective_tf = m.result().numpy()
        ## end TF operations ##        

        objective = mse_objective # maximize
        return objective
    return roc_objective_val

def load_model(model_name, custom_objects={'QDense': QDense, 'QActivation': QActivation}):
    name = model_name + '.json'
    json_file = open(name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects=custom_objects)
    model.load_weights(model_name + '.h5')
    return model

def save_model(model_save_name, model):
    with open(model_save_name + '.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save_weights(model_save_name + '.h5')
