# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:10:53 2019

@author: user
"""

def flip_prob_wrapper(input_tensor,random_matrix):
    x = input_tensor
    R = random_matrix
    def flip_probability(y_true,y_pred):
        hi = classifier.layers[0].get_weights()[0]
#       layer_name = "output"
#       intermediate_layer_model = Model(inputs=classifier.input,outputs=classifier.get_layer(layer_name).output)
#       intermediate_output = intermediate_layer_model.predict(X)
        hi_x = kb.dot(kb.transpose(hi),kb.transpose(x))
        hi_R = kb.dot(kb.transpose(hi),kb.transpose(R))
        R_x = kb.dot(R,kb.transpose(x)) # randomly projecting down data
        hi_R_R_x = kb.dot(hi_R,R_x)
        
        # getting either 1 or 0 in a differentiable (?) way
        #signs = hi_R_R_x*hi_x
        #signs = kb.maximum(signs,0) # sets negative values to zero
        #signs = kb.softmax(signs) # reduces positive values to below one
        #signs = kb.maximum(signs,1) # sets positive values to 1
       
    #flips = kb.sum(signs)
#                
        projected_signs =  kb.sign(hi_R_R_x) #kb.softsign(hi_R_R_x)
        original_signs =  kb.sign(hi_x) #kb.softsign(hi_x)
        
        flips = kb.sum(kb.cast(kb.equal(projected_signs,original_signs),dtype = "float32"))
        size = (kb.int_shape(hi_R_R_x)[0]*kb.int_shape(hi_R_R_x)[1])
        fake_loss = (y_true - y_pred) * 0
        return fake_loss + flips/size
    return flip_probability