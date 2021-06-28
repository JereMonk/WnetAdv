
import tensorflow as tf
from PIL import Image 
import numpy as np
import PIL 


def rescale(image):
    return( np.array(((image+1)/2)*255 ).astype("uint8") )

def set_learning_rate(step_counter,model,base_lr,steps,decay_step,decay_rate):
    
    if(step_counter<=decay_step):
        new_lr = base_lr
    
    else:
        new_lr = base_lr**(decay_rate*(step_counter//decay_step))
    model.optimizer.lr = new_lr
    
def training(model,train_dataset,test_dataset,max_iter,start_iter,base_lr,ckpt_freq,img_freq,dir_path,solver_steps,test_freq,reconstruction_loss_weight,decay_step,decay_rate,sigma,blur_kernel,noise_amp):
  
    ##TRAIN
    total_train_loss_encoder = []
    total_train_loss_decoder = []

    step_counter=start_iter
    writer = tf.summary.create_file_writer(dir_path)
    
 
    while(step_counter<max_iter):
        


        print("\nStart of iter %d" % (step_counter,))
        print("Learning rate" +str(model.optimizer.lr))


        

        for _, x_batch_train in enumerate(train_dataset):
            step_counter+=1
            set_learning_rate(step_counter,model,base_lr,solver_steps,decay_step=decay_step,decay_rate=decay_rate)

            train_losses = model.train_step(x_batch_train,reconstruction_loss_weight,sigma,blur_kernel,noise_amp)

            train_losses_encoder=train_losses["loss_encoder"]
            train_losses_decoder=train_losses["loss_decoder"]

            total_train_loss_encoder.append(train_losses_encoder.numpy())
            total_train_loss_decoder.append(train_losses_decoder.numpy())

            if step_counter%ckpt_freq ==0:
                model.save_weights(dir_path+"/ckpt"+str(step_counter))
            
            if step_counter%1==0:
                final_train_loss_decoder = np.mean(np.array(total_train_loss_decoder))
                final_train_loss_encoder = np.mean(np.array(total_train_loss_encoder))
                
                print("step "+str(step_counter) ) 
                print("TRAIN LOSS ENCODER : ", final_train_loss_encoder)
                print("TRAIN LOSS DECODER : ", final_train_loss_decoder)
                total_train_loss_encoder = []
                total_train_loss_decoder = []

                with writer.as_default():
                    tf.summary.scalar('training loss encoder', final_train_loss_encoder, step=step_counter)
                    tf.summary.scalar('training loss decoder', final_train_loss_decoder, step=step_counter)
                    tf.summary.scalar('training loss', final_train_loss_encoder+final_train_loss_decoder, step=step_counter)
                    tf.summary.scalar('learning rate',model.optimizer.lr , step=step_counter)

            if step_counter%img_freq==0:
                image_ref= tf.expand_dims(x_batch_train[0],0)

                img = rescale(x_batch_train[0])
                Image.fromarray(img).save(dir_path+"/image_step"+ str(step_counter)+"_.png")

                res = model(image_ref).numpy()[0]
                res = rescale(res)
                Image.fromarray(res).save(dir_path+"/reconstruction_step"+ str(step_counter)+"_.png")

                seg = model.encoder(image_ref)
                ag = tf.math.argmax(seg, axis=-1, output_type=tf.dtypes.int64)[0]
                ag = ag.numpy()
                ag = ag *255 /ag.max()
                Image.fromarray(ag).convert("L").save(dir_path+"/segmentation_step"+ str(step_counter)+"_.png")



            if step_counter%test_freq==0:

                print('START TEST')
                total_test_loss_encoder = []
                total_test_loss_decoder = []

                for _, x_batch_test in enumerate(test_dataset):

                    test_losses = model.test_step(x_batch_test,reconstruction_loss_weight)
                    test_losses_encoder=test_losses["loss_encoder"]
                    test_losses_decoder=test_losses["loss_decoder"]

                    total_test_loss_encoder.append(test_losses_encoder)
                    total_test_loss_decoder.append(test_losses_decoder)

                final_test_loss_decoder = np.mean(np.array(total_test_loss_decoder.numpy()))
                final_test_loss_encoder = np.mean(np.array(total_test_loss_encoder.numpy()))

                print("TEST LOSS ENCODER : ", final_test_loss_encoder)
                print("TEST LOSS DECODER : ", final_test_loss_decoder)

                with writer.as_default():
                    tf.summary.scalar('test loss encoder', final_test_loss_encoder, step=step_counter)
                    tf.summary.scalar('test loss decoder', final_test_loss_decoder, step=step_counter)
                    tf.summary.scalar('test loss', final_test_loss_decoder+final_test_loss_encoder, step=step_counter)
