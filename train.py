def build_model(input_shape =(10,96,96,96) ,output_channels):
    c, H, W, D = input_shape
    inp = Input(input_shape)
    
    x0 = light_green_block(inp)
    x1= dark_green_block(x0,64)
    x2 = dark_green_block(x1, 128)
    x3 = dark_green_block(x2, 256)
    x4 = dark_green_block(x3,512)
    #######################
    ## Code for the latent dimensions layer
    # To be added later by Ashwin
    
    #######################
    x = blue_block(x4)
    # Skip connection
    x = Add(name='Input_Dec_GT_512')([x, x4])
    x = blue_block(x)
    x = Add(name = 'Input_Dec_GT_256')([x, x3])
    x = blue_block(x)
    x = Add(name = 'Input_Dec_GT_128')([x, x2])
    x = blue_block(x)
    x = Add(name = 'Input_Dec_GT_64')([x, x1])
    x = blue_block(x)
    output = Add(name = 'Input_Dec_GT_32')([x , x0])
    

    model = Model(inp, output)
    model.compile(
        Adam(learning_rate = 1e-4), # to be customized
        custom_loss(),# call the loss function here
        metrics=[dice_coefficient] # to be customized
    )
    return model
  
  
  def custom_loss():
    L_theta =(1/D).* np.exp(-s_i ).* np.mod(y_i - y_i_cap) + s_i
    return L_theta

def uncertatinity_estimate(s_i):
    return 2.* np.exp(2 .* s_i)


# def make_batches_from_data(X_train):

def dice_coefficient(y,y_hat):
