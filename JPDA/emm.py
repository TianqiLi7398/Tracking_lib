track_list = []
track_list_next_index = []
t = 0

jpda_est = []
ellips_inputs = []
text_list = []

init_time = time.time()



for seq in range(len(obs_list)):
    

    t += dt
    
    next_index_list = []
    jpda_k_est = []
    ellips_inputs_k = []
    text_list_k = []
    # 1. extract all obs inside elliposidual gates of initialized and confirmed tracks

    obs_matrix_list = []
    
    for single_sensor_obs in multi_sensor_obs:
        z_k = single_sensor_obs[seq]
        # print(len(track_list_next_index), len(z_k))
        obs_matrix = np.zeros((len(track_list_next_index), len(z_k)))
        obs_matrix_list.append(obs_matrix)

    for i in range(len(track_list_next_index)):
        k = track_list_next_index[i]
        kf = track_list[k].kf
        S = np.dot(np.dot(kf.H, kf.P_k_k_min), kf.H.T) + kf.R
        pred_z = np.dot(kf.H, kf.x_k_k_min)
        
        #### usage for plotting ellipse
        track_list[k].S = S
        track_list[k].pred_z = pred_z

        for l in range(len(multi_sensor_obs)):
            z_k = multi_sensor_obs[l][seq]
            
            for j in range(len(z_k)):
                z = z_k[j]
                z_til = pred_z - np.matrix(z).T
                
                obs_matrix_list[l][i][j] = elliposidualGating(z_til, S)
            
    print(obs_matrix_list)
    # create a mitrix to check the relationship between obs and measurements

    for l in range(len(obs_matrix_list)):
        # multisensor change
        obs_matrix = obs_matrix_list[l]
        z_k = multi_sensor_obs[l][seq]

        obs_sum = obs_matrix.sum(axis=0)
        # matrix operation, exclude all observations whose sum is 0, which means its not in any gate
        index = np.where(obs_sum > 0)[0]
        obs_outside_index = np.where(obs_sum == 0)[0]
        print(seq, l, obs_outside_index)
        

        # 2. deal with observations inside all gates
        # i. initialize each observation, find out how many gates each observation is in
        
        obs_class = []
        # now in obs_class, each element is a class with its .track includes all 
        # track numbers from 0 - m_k
        # then we need to analyse the gate. First for jth obs z_j in its ith gate we 
        # need to calculate
        # g_ij = N(z_j - z_ij|0, S_ij) here z_ij and S_ij is the updated KF given 
        # observation z_j

        
        for i in range(len(z_k)):

            if i in index:
                try:
                    a_mea = measurement(z_k[i], i)
                except:
                    print(i, z_k)
                for j in np.where(obs_matrix[:, i] != 0)[0]:
                    # track is the track id for obs i, in obs_matrix the column value is not 0
                    track_id = track_list_next_index[j]
                    a_mea.inside_track(track_id)
                    temp_kf = copy.deepcopy(track_list[track_id].kf)
                    temp_kf.update(z_k[i])
                    var = multivariate_normal(mean=np.squeeze(temp_kf.z_bar.reshape(2).tolist()[0]), cov=temp_kf.S_k)
                    a_mea.g_ij.append(var.pdf(z_k[i]))
                obs_class.append(a_mea)
            else:
                obs_class.append([])
        
        # ii. for each gate/track, analysis if there exists observations joint 
        # different tracks, find out the relation between different tracks
        
        
        track_class = []
        for i in range(len(track_list_next_index)):
            k = track_list_next_index[i]
            a_track = track_list[k]   # pointer to track_list class
            a_track.get_measurement(np.where(obs_matrix[i] != 0)[0])
            a_track.measurement.append(-1)
            track_class.append(a_track)



        for i in range(len(track_class)):
            
            # Case 1. if there is no obs inside track gate, 
            # make Kalman's update x_k_k= x_k_k_min, P_k_k = P_k_k_min
            if track_class[i].deleted:
                # too early to delete
                continue
            
            if len(track_class[i].measurement) == 1:
                kf = track_class[i].kf
                # apparently this is wrong, we need to make observation in Kalman update 
                # x_k_k_min, then update the covariance.
                kf.x_k_k = kf.x_k_k_min
                kf.P_k_k = kf.P_k_k_min
                #  kf.update([kf.x_k_k_min[0, 0], kf.x_k_k_min[1, 0]])
                
                
                track_class[i].update(t, kf, False, l)

                # only keep the confirmed ones
                if track_class[i].confirmed:
                    jpda_k_est.append(track_class[i].record["points"][-1])
                    #  ellips_inputs_k.append([track_class[i].S, track_class[i].pred_z])
                    ellips_inputs_k.append([track_class[i].kf.P_k_k[0:2, 0:2], track_class[i].kf.x_k_k[0:2, 0]])
                    text_list_k.append([track_class[i].kf.x_k_k[0:2, 0], False])
                if not (track_class[i].deleted or track_class[i].abandoned):
                    next_index_list.append(track_list_next_index[i])
                continue;
            
            # Case 2. there is obs inside the track gate
            # calculate the beta for ith track
            # need the number of measurements inside the gate
            beta = cal_beta(len(track_class[i].measurement) - 1)
      
            table_key = [track_list_next_index[i]]
            # begin find all observations inside this gate that is related to other gates (joint)
            
            for obs_id in track_class[i].measurement:
                if obs_id != -1:
                    obs = obs_class[obs_id]
                    table_key += obs.track
                
            table_key = list(set(table_key))
            
            # for each track, figure out how many observations inside the track
            
            # inverse the table
            table_key_inv = []
            for j in table_key:
                table_key_inv.append(track_list_next_index.index(j))
            
            table_key_matrix = obs_matrix[table_key_inv]
            

            # there are overlaps of tracks, we only remain the one has oldest history
            if (table_key_matrix == table_key_matrix[0]).all() and len(table_key_matrix) > 2:
                # there are overlaps of tracks, we only remain the one has oldest history
                seed = min(table_key)
                for key in table_key:
                    if key != seed:
                        track_list[key].deletion(t)
            
            # in order to avoid multiple tracks tracking same target, we need to analysis the
            # obs_matrix
            
            obs_num_tracks = obs_matrix.sum(axis=1)[table_key_inv]

            # number of joint tracks
            N_T = len(table_key)
            # number of observations total
            total_obs = []
            for track_id in table_key:
                a_track = track_list[track_id]
                total_obs +=a_track.measurement

            total_obs = list(set(total_obs))
            N_o = len(total_obs) - 1

            common_factor = common_fact(beta, P_D, N_T, N_o)

            # iii. after merged all related tracks, we generat a hypothesis matrix/table based on the table 
            # generated by the table_key
            obs_num_tracks_ = obs_num_tracks + 1
            total_row = int(obs_num_tracks_.prod())

            # create title for the table


            hyp_matrix = {}
            for a_key in table_key:
                hyp_matrix[str(a_key)] = []
            hyp_matrix["p"] = []
            
            for row_num in range(total_row):
                key_num = len(table_key)
                col_num = 0
                # build one row of hypothesis
                while key_num > 0:
                    if col_num == len(table_key) - 1:
                        obs_id = int(row_num)
                        product = 1
                    else:
                        product = obs_num_tracks_[(col_num + 1):].prod()
                        obs_id = int(row_num // product)
                    
                    
                    value = track_list[table_key[col_num]].measurement[obs_id]
                    
                    key = str(table_key[col_num])
                    hyp_matrix[key].append(value)
                    row_num = row_num % product
                    col_num += 1
                    key_num -= 1
                
                # now we want to calculate the probability of this row's hypothesis
                hyp_list = []

                prob = common_factor
                for key in hyp_matrix.keys():
                    if key != 'p':
                        hyp_list.append(hyp_matrix[key][-1])
                
                # print('hyp_list, ', hyp_list)
                # calculate the prob of this hypothesis
                if checkIfDuplicates(hyp_list):
                    # this is not one vaild hypothesis.
                    prob = 0
                else:
                    # this is the valid hypothesis, we should calculate the prob
                    for key in hyp_matrix.keys():
                        if key != 'p':
                            track_id = int(key)
                            obs_id = hyp_matrix[key][-1]
                            # print('obs id ', obs_id)
                            if obs_id == -1:
                                prob *= (1 - P_D) * beta
                            else:
                                # print(obs_class[obs_id].table, print(obs_class[obs_id].id))
                                index = obs_class[obs_id].track.index(track_id)
                                prob *= P_D * obs_class[obs_id].g_ij[index]
                hyp_matrix['p'].append(prob)
                
            # iv. Then gather the prob in this track, and update the kF
            obs_in_i_track = track_class[i].measurement
            obs_in_i_track_prob = []
            hyp_in_i_track = np.array(hyp_matrix[str(track_list_next_index[i])])
            hyp_in_i_track_prob = np.array(hyp_matrix['p'])
            for obs in obs_in_i_track:
                index_ = np.where(hyp_in_i_track == obs)
                w_ij_list = hyp_in_i_track_prob[index_]
                obs_in_i_track_prob.append(w_ij_list.sum())
            
            # then normalize all w_ij s
            obs_in_i_track_prob_norm = normalize(obs_in_i_track_prob)

            # well, we then just need to update the KF of ith track
            kf = track_class[i].kf
            x_k = []
            P_k = []

            for obs in obs_in_i_track:
                if obs == -1:
                    x_k.append(kf.x_k_k)
                    P_k.append(kf.P_k_k)
                else:
                    
                    z = np.array(z_k[obs]).T

                    # update the kf
                    temp_kf = copy.deepcopy(kf)

                    temp_kf.update(z)
                    
                    x_k.append(temp_kf.x_k_k)
                    P_k.append(temp_kf.P_k_k)
            
            x_k_pda = 0 * temp_kf.x_k_k
            P_k_pda = 0 * temp_kf.P_k_k
            for j in range(len(obs_in_i_track_prob_norm)):
                x_k_pda += obs_in_i_track_prob_norm[j] * x_k[j]

            for j in range(len(obs_in_i_track_prob_norm)):
                P_k_pda += obs_in_i_track_prob_norm[j] * (P_k[j] + np.dot(x_k_pda - x_k[j], x_k_pda.T - x_k[j].T))

            # update this to kf
            kf.x_k_k = x_k_pda
            kf.P_k_k = P_k_pda
            
            # Too large the uncertainty
            if np.linalg.det(kf.P_k_k[0:2, 0:2]) > common_P:
                track_class[i].update(t, kf, False, l)
                continue
            
            
            track_class[i].update(t, kf, True, l)
            
            # only draw/output the confirmed ones
            # if track_class[i].confirmed:

            #     jpda_k_est.append(x_k_pda)
            #     ellips_inputs_k.append([track_class[i].kf.P_k_k[0:2, 0:2], track_class[i].kf.x_k_k[0:2, 0]])
            #     text_list_k.append([track_class[i].kf.x_k_k[0:2, 0], True])
                
            # save the activated ones for next recursion
            if not (track_class[i].deleted or track_class[i].abandoned):
                next_index_list.append(track_list_next_index[i])
        # 3. deal with observations outside all gates

        # now initialize all observations outside of the gate
        for i in obs_outside_index:
            z = z_k[i] + [0, 0]
            x0 = np.matrix(z).T
            kf = EKFcontrol(F, H, x0, P, Q, R)
            id_ = len(track_list)
            new_track = track(t, id_, kf, len(multi_sensor_obs))
            new_track.kf.predict()
            track_list.append(new_track)
            next_index_list.append(id_)

    # keep the record
    for i in range(len(track_list_next_index)):
        k = track_list_next_index[i]
        if track_list[k].confirmed:
            jpda_k_est.append(track_list[k].kf.x_k_k)
            ellips_inputs_k.append([track_list[k].kf.P_k_k[0:2, 0:2], track_list[k].kf.x_k_k[0:2, 0]])
                

    track_list_next_index = list(set(next_index_list))
    
    jpda_est.append(jpda_k_est)
    ellips_inputs.append(ellips_inputs_k)
    text_list.append(text_list_k)
    


print(time.time() - init_time)

