def verify(pred_tensor,target_tensor):

    print(target_tensor.size())
    print(pred_tensor.size())

    print(pred_tensor[0,0,0,:])

    '''object_mask_target = (target_tensor[:,:,:,4]>0) | (target_tensor[:,:,:,9]>0) 
    object_mask_pred = (pred_tensor[:,:,:,4]>0) | (pred_tensor[:,:,:,9]>0) 

    print(target_tensor[object_mask_target])
    print(pred_tensor[object_mask_target])
    print(object_mask_pred)'''

    import sys
    sys.exit(0)