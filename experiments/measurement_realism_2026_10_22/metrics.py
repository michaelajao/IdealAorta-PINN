"""Hidden-target temporal diagnostics; never used for fitting or model selection."""
import numpy as np


def temporal_error_decomposition(predicted_pair,truth_pair,times_s):
    prediction=np.asarray(predicted_pair,dtype=float)
    truth=np.asarray(truth_pair,dtype=float)
    times=np.asarray(times_s,dtype=float)
    if prediction.ndim!=3 or prediction.shape[0]!=2 or prediction.shape[2]!=3 or truth.shape!=prediction.shape:
        raise ValueError('matching (2,N,3) prediction/truth pairs required')
    if len(prediction[0])==0 or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError('finite nonempty velocity arrays required')
    if times.shape!=(2,) or not np.isfinite(times).all() or times[0]==times[1]:
        raise ValueError('two finite distinct times required')
    dt=float(times[1]-times[0])
    errors=prediction-truth
    means=errors.mean(axis=1)
    centered=errors-means[:,None]
    second=np.mean(errors*errors,axis=1)
    cross=np.mean(errors[0]*errors[1],axis=0)
    covariance=np.mean(centered[0]*centered[1],axis=0)
    variance=np.mean(centered*centered,axis=1)
    correlation=np.divide(covariance,np.sqrt(variance[0]*variance[1]),
                          out=np.zeros(3),where=(variance[0]*variance[1])>1e-30)
    acceleration_error=(errors[1]-errors[0])/dt
    direct=np.mean(acceleration_error**2,axis=0)
    reconstructed=(second[0]+second[1]-2*cross)/dt**2
    bias=(means[1]-means[0])/dt
    return {'delta_t_s':dt,'snapshot_error_bias_m_s':means.tolist(),
            'snapshot_error_component_mse_m2_s2':second.tolist(),
            'cross_time_error_second_moment_m2_s2':cross.tolist(),
            'cross_time_error_covariance_m2_s2':covariance.tolist(),
            'cross_time_error_component_correlation':[float(correlation[c]) if variance[0,c]*variance[1,c]>1e-30 else None for c in range(3)],
            'acceleration_error_bias_m_s2':bias.tolist(),
            'acceleration_error_component_mse_m2_s4':direct.tolist(),
            'acceleration_error_rms_vector_m_s2':float(np.sqrt(direct.sum())),
            'acceleration_error_bias_squared_m2_s4':float(np.sum(bias**2)),
            'acceleration_error_variance_sum_m2_s4':float(np.sum(np.var(acceleration_error,axis=0))),
            'second_moment_identity_max_abs_error_m2_s4':float(np.max(np.abs(direct-reconstructed))),
            'interpretation':'post hoc error decomposition, not proof of a physiological mechanism'}
