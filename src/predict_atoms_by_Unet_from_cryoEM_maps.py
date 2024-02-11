import numpy as np
import copy

def pad_map_data(x,unet_depth):
    if x.ndim == 3:
        x = np.expand_dims(x,axis=-1)
    elif x.ndim == 4:
        assert x.shape[-1] == 1, print("The last channel should be 1D.")
    elif x.ndim != 4:
        raise VauleError("The input map should be 3D or 4D data!")
            
    const_num = 2**(unet_depth-1)
    shape0, shape1, shape2 = x.shape[0:3]
    if shape0 % const_num == 0:
        padding0 = 0
    else:
        padding0 = (const_num - shape0 % const_num)
    if shape1 % const_num == 0:
        padding1 = 0
    else:
        padding1 = (const_num - shape1 % const_num) 
    if shape2 % const_num == 0:
        padding2 = 0
    else:
        padding2 = (const_num - shape2 % const_num)
    if (padding0 == 0) and (padding1 == 0) and (padding2 == 0):
        return x
    pad_width = ((0,padding0),(0,padding1),(0,padding2),(0,0))
    x = np.pad(x,pad_width)
    #print(f"Map data has been padded from ({shape0},{shape1},{shape2}) to {x.shape[0:3]}.")
    return x

def slice_large_map(x,patch_size):
    if x.ndim == 3:
        x = np.expand_dims(x,axis=-1)
    elif x.ndim != 4:
        raise VauleError("The input map should be 3D or 4D data!")

    shape0, shape1, shape2 = x.shape[0:3]
    if shape0 > patch_size:
        ni = np.ceil((shape0-patch_size)/(patch_size-32))
        ni = int(ni) + 1
    else:
        ni = 1

    if shape1 > patch_size:
        nj = np.ceil((shape1-patch_size)/(patch_size-32))
        nj = int(nj) + 1
    else:
        nj = 1

    if shape2 > patch_size:
        nk = np.ceil((shape2-patch_size)/(patch_size-32))
        nk = int(nk) + 1
    else:
        nk = 1

    list_patches = []
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                ib = (patch_size-32)*i
                ie = ib + patch_size
                if ie > shape0:
                    ie = shape0
                jb = (patch_size-32)*j
                je = jb + patch_size
                if je > shape1:
                    je = shape1
                kb = (patch_size-32)*k
                ke = kb + patch_size
                if ke > shape2:
                    ke = shape2
                patch = x[ib:ie,jb:je,kb:ke]
                patch = pad_map_data(patch,5)
                index = (ib,ie,jb,je,kb,ke)
                list_patches.append([patch,index])

    return list_patches

def combine_sliced_prediction(map_shape,list_predatoms_indices):
    pred_atoms = np.zeros(map_shape,dtype=list_predatoms_indices[0][0].dtype)
    for atoms_index in list_predatoms_indices:
        atoms, index = atoms_index
        ib, ie, jb, je, kb, ke = index
        if ib != 0:
            ib = ib + 16
            ib2 = 16
        else:
            ib2 = 0
        if ie != map_shape[0]:
            ie = ie - 16
            ie2 = atoms.shape[0] - 16
        else:
            ie2 = ib2 + (ie - ib)

        if jb != 0:
            jb = jb + 16
            jb2 = 16
        else:
            jb2 = 0
        if je != map_shape[1]:
            je = je - 16
            je2 = atoms.shape[1] - 16
        else:
            je2 = jb2 + (je - jb)

        if kb != 0:
            kb = kb + 16
            kb2 = 16
        else:
            kb2 = 0
        if ke != map_shape[2]:
            ke = ke - 16
            ke2 = atoms.shape[2] - 16
        else:
            ke2 = kb2 + (ke - kb)
        pred_atoms[ib:ie,jb:je,kb:ke] = atoms[ib2:ie2,jb2:je2,kb2:ke2]

    return pred_atoms

def predict_atoms_from_unet(model,map_data,map_shape,unet_depth,patch_size):
    import tensorflow as tf

    if map_shape[0]*map_shape[1]*map_shape[2] <= patch_size*patch_size*patch_size:
        padded_map_data = pad_map_data(map_data,unet_depth)
        if padded_map_data.ndim == 4:
            padded_map_data = np.expand_dims(padded_map_data,axis=0)
        else:
            raise ValueError("Wrong input map data shape.")

        y_pred = model.predict(padded_map_data,batch_size=1)
        y_pred = np.squeeze(y_pred,axis=0)
        y_pred_atoms = np.squeeze((np.argmax(y_pred,axis=-1)).astype(np.int8))
        y_pred_atoms_onehot = tf.keras.utils.to_categorical(y_pred_atoms,num_classes=19)
        y_pred_prob = y_pred * y_pred_atoms_onehot
        y_pred_prob = np.sum(y_pred_prob,axis=-1)
        y_pred_atoms = y_pred_atoms[0:map_shape[0],0:map_shape[1],0:map_shape[2]]
        y_pred_prob = y_pred_prob[0:map_shape[0],0:map_shape[1],0:map_shape[2]]
        return y_pred_atoms, y_pred_prob
    else:
        list_patches_indices = slice_large_map(map_data,patch_size)
        list_pred_atoms_indices = []
        list_pred_prob_indices = []
        for patch_index in list_patches_indices:
            patch, index = patch_index
            if patch.ndim == 4:
                patch = np.expand_dims(patch,axis=0)
            elif patch.ndim != 5:
                raise ValueError("Wrong input map data shape")
            y_pred = model.predict(patch,batch_size=1)
            y_pred = np.squeeze(y_pred,axis=0)
            y_pred_atoms = np.squeeze(np.argmax(y_pred,axis=-1).astype(np.int8))
            y_pred_atoms_onehot = tf.keras.utils.to_categorical(y_pred_atoms,num_classes=19)
            y_pred_prob = y_pred * y_pred_atoms_onehot
            y_pred_prob = np.sum(y_pred_prob,axis=-1)
            list_pred_atoms_indices.append([y_pred_atoms,index])
            list_pred_prob_indices.append([y_pred_prob,index])
        full_pred_atoms = combine_sliced_prediction(map_shape,list_pred_atoms_indices)
        full_pred_prob = combine_sliced_prediction(map_shape,list_pred_prob_indices)
        return full_pred_atoms, full_pred_prob

def convert_map_to_atoms_by_two_patch_sizes(model_name,map_data_raw,contour,patch_size1,patch_size2):
    from tensorflow.keras.models import load_model

    unet_depth = 5
    model = load_model(model_name,compile=False)
    #model.summary(positions=[0.2,0.7,0.8,1])

    if contour < 0.:
        new_contour = contour
    else:
        new_contour = contour*0.5

    map_data = copy.deepcopy(map_data_raw)

    percentile = 95.0
    percentile_value = np.percentile(map_data[map_data>=new_contour],percentile)
    map_data[map_data>percentile_value] = percentile_value # truncate to 95 percentile
    map_data -= new_contour
    map_data[map_data<0.0] = 0.0
    map_data /= (percentile_value-new_contour) # normalize to [0.0,1.0]
    #print(f"Map data has been truncated and normalized.")
    #print(f"Min: {np.min(map_data)}, Max: {np.max(map_data)}, Mean: {np.mean(map_data)}, Median: {np.median(map_data)}")

    map_shape = np.squeeze(map_data).shape
    assert len(map_shape) == 3, print("3D map data is needed.")

    full_pred_atoms1, full_pred_prob1 = predict_atoms_from_unet(model,map_data,map_shape,unet_depth,patch_size1)

    full_pred_atoms2, full_pred_prob2 = predict_atoms_from_unet(model,map_data,map_shape,unet_depth,patch_size2)

    return full_pred_atoms1, full_pred_prob1, full_pred_atoms2, full_pred_prob2
